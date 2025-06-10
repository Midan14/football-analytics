import { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import { useAuth } from '../context/AuthContext';

// WebSocket configuration
const WS_CONFIG = {
  baseURL: process.env.REACT_APP_WS_URL || 'ws://localhost:3001',
  reconnectInterval: 3000, // 3 seconds
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000, // 30 seconds
  connectionTimeout: 10000, // 10 seconds
  protocols: ['football-analytics-v1'],
  enableHeartbeat: true,
  enableAutoReconnect: true,
  enableLogging: process.env.NODE_ENV === 'development'
};

// Message types for different real-time events
export const MESSAGE_TYPES = {
  // Authentication
  AUTH: 'auth',
  AUTH_SUCCESS: 'auth_success',
  AUTH_FAILURE: 'auth_failure',
  
  // Match Events
  MATCH_START: 'match_start',
  MATCH_END: 'match_end',
  MATCH_UPDATE: 'match_update',
  GOAL_SCORED: 'goal_scored',
  CARD_ISSUED: 'card_issued',
  SUBSTITUTION: 'substitution',
  LIVE_STATS: 'live_stats',
  
  // Predictions
  PREDICTION_UPDATE: 'prediction_update',
  PREDICTION_RESULT: 'prediction_result',
  MODEL_UPDATE: 'model_update',
  
  // Injuries
  INJURY_REPORTED: 'injury_reported',
  INJURY_UPDATE: 'injury_update',
  RECOVERY_UPDATE: 'recovery_update',
  
  // User Events
  FAVORITE_ADDED: 'favorite_added',
  FAVORITE_REMOVED: 'favorite_removed',
  USER_NOTIFICATION: 'user_notification',
  
  // System Events
  HEARTBEAT: 'heartbeat',
  PING: 'ping',
  PONG: 'pong',
  SYSTEM_MESSAGE: 'system_message',
  ERROR: 'error',
  
  // Subscriptions
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  SUBSCRIPTION_SUCCESS: 'subscription_success',
  SUBSCRIPTION_ERROR: 'subscription_error'
};

// Subscription channels
export const CHANNELS = {
  MATCHES_LIVE: 'matches:live',
  MATCH_DETAIL: 'match:detail:',
  TEAM_UPDATES: 'team:updates:',
  PLAYER_UPDATES: 'player:updates:',
  LEAGUE_UPDATES: 'league:updates:',
  PREDICTIONS: 'predictions',
  INJURIES: 'injuries',
  USER_NOTIFICATIONS: 'user:notifications:',
  SYSTEM_ALERTS: 'system:alerts'
};

// Connection states
export const CONNECTION_STATE = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  ERROR: 'error'
};

// Main WebSocket hook
export const useWebSocket = (options = {}) => {
  const { isAuthenticated, getAuthHeaders, user } = useAuth();
  const { actions: appActions, state: appState } = useApp();
  
  // Configuration with defaults
  const config = {
    autoConnect: true,
    enableHeartbeat: WS_CONFIG.enableHeartbeat,
    enableAutoReconnect: WS_CONFIG.enableAutoReconnect,
    enableLogging: WS_CONFIG.enableLogging,
    onMessage: null,
    onConnect: null,
    onDisconnect: null,
    onError: null,
    onReconnect: null,
    ...options
  };
  
  // State management
  const [connectionState, setConnectionState] = useState(CONNECTION_STATE.DISCONNECTED);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [subscriptions, setSubscriptions] = useState(new Set());
  const [messageHistory, setMessageHistory] = useState([]);
  
  // Refs for management
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const heartbeatIntervalRef = useRef(null);
  const connectionTimeoutRef = useRef(null);
  const messageCallbacksRef = useRef(new Map());
  const mountedRef = useRef(true);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, []);
  
  // Log function
  const log = useCallback((level, message, data = null) => {
    if (config.enableLogging) {
      console[level](`[WebSocket] ${message}`, data || '');
    }
  }, [config.enableLogging]);
  
  // Generate WebSocket URL with auth params
  const getWebSocketURL = useCallback(() => {
    let url = WS_CONFIG.baseURL;
    
    if (isAuthenticated && user) {
      const params = new URLSearchParams({
        userId: user.id,
        sessionId: Date.now().toString()
      });
      url += `?${params.toString()}`;
    }
    
    return url;
  }, [isAuthenticated, user]);
  
  // Send message to WebSocket
  const sendMessage = useCallback((type, data = {}, channel = null) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      log('warn', 'Cannot send message - WebSocket not connected');
      return false;
    }
    
    const message = {
      type,
      data,
      channel,
      timestamp: new Date().toISOString(),
      userId: user?.id || null
    };
    
    try {
      wsRef.current.send(JSON.stringify(message));
      log('debug', `Sent message: ${type}`, message);
      return true;
    } catch (error) {
      log('error', 'Failed to send message', error);
      setError(error.message);
      return false;
    }
  }, [user, log]);
  
  // Handle incoming messages
  const handleMessage = useCallback((event) => {
    try {
      const message = JSON.parse(event.data);
      log('debug', `Received message: ${message.type}`, message);
      
      if (!mountedRef.current) return;
      
      setLastMessage(message);
      
      // Add to message history (keep last 100 messages)
      setMessageHistory(prev => [
        message,
        ...prev.slice(0, 99)
      ]);
      
      // Handle system messages
      switch (message.type) {
        case MESSAGE_TYPES.AUTH_SUCCESS:
          log('info', 'Authentication successful');
          setError(null);
          break;
          
        case MESSAGE_TYPES.AUTH_FAILURE:
          log('error', 'Authentication failed', message.data);
          setError(message.data?.message || 'Authentication failed');
          break;
          
        case MESSAGE_TYPES.PONG:
          log('debug', 'Received pong');
          break;
          
        case MESSAGE_TYPES.ERROR:
          log('error', 'Server error', message.data);
          setError(message.data?.message || 'Server error');
          appActions.addNotification({
            type: 'error',
            title: 'WebSocket Error',
            message: message.data?.message || 'An error occurred'
          });
          break;
          
        case MESSAGE_TYPES.SYSTEM_MESSAGE:
          log('info', 'System message', message.data);
          appActions.addNotification({
            type: 'info',
            title: 'System Message',
            message: message.data?.message
          });
          break;
          
        case MESSAGE_TYPES.SUBSCRIPTION_SUCCESS:
          log('info', `Subscribed to: ${message.data?.channel}`);
          break;
          
        case MESSAGE_TYPES.SUBSCRIPTION_ERROR:
          log('error', `Subscription failed: ${message.data?.channel}`, message.data);
          break;
          
        // Match events
        case MESSAGE_TYPES.MATCH_START:
        case MESSAGE_TYPES.MATCH_UPDATE:
        case MESSAGE_TYPES.GOAL_SCORED:
        case MESSAGE_TYPES.CARD_ISSUED:
        case MESSAGE_TYPES.SUBSTITUTION:
          // Trigger match list refresh
          window.dispatchEvent(new CustomEvent('match-update', {
            detail: message
          }));
          break;
          
        case MESSAGE_TYPES.MATCH_END:
          appActions.addNotification({
            type: 'info',
            title: 'Match Ended',
            message: `${message.data?.homeTeam} vs ${message.data?.awayTeam} has ended`
          });
          break;
          
        // Prediction events
        case MESSAGE_TYPES.PREDICTION_UPDATE:
        case MESSAGE_TYPES.PREDICTION_RESULT:
          window.dispatchEvent(new CustomEvent('prediction-update', {
            detail: message
          }));
          break;
          
        // Injury events
        case MESSAGE_TYPES.INJURY_REPORTED:
          appActions.addNotification({
            type: 'warning',
            title: 'Injury Reported',
            message: `${message.data?.playerName} - ${message.data?.injury}`
          });
          window.dispatchEvent(new CustomEvent('injury-update', {
            detail: message
          }));
          break;
          
        case MESSAGE_TYPES.RECOVERY_UPDATE:
          window.dispatchEvent(new CustomEvent('injury-update', {
            detail: message
          }));
          break;
          
        // User events
        case MESSAGE_TYPES.USER_NOTIFICATION:
          appActions.addNotification({
            type: message.data?.type || 'info',
            title: message.data?.title || 'Notification',
            message: message.data?.message
          });
          break;
          
        default:
          log('debug', `Unhandled message type: ${message.type}`);
      }
      
      // Call registered callbacks for specific message types
      const callbacks = messageCallbacksRef.current.get(message.type) || [];
      callbacks.forEach(callback => {
        try {
          callback(message);
        } catch (error) {
          log('error', 'Message callback error', error);
        }
      });
      
      // Call global message handler
      if (config.onMessage) {
        config.onMessage(message);
      }
      
    } catch (error) {
      log('error', 'Failed to parse message', error);
      setError('Failed to parse server message');
    }
  }, [log, config.onMessage, appActions]);
  
  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (!config.enableHeartbeat) return;
    
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        sendMessage(MESSAGE_TYPES.PING);
      }
    }, WS_CONFIG.heartbeatInterval);
    
    log('debug', 'Heartbeat started');
  }, [config.enableHeartbeat, sendMessage, log]);
  
  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
      log('debug', 'Heartbeat stopped');
    }
  }, [log]);
  
  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      log('warn', 'Already connected');
      return;
    }
    
    setConnectionState(CONNECTION_STATE.CONNECTING);
    setError(null);
    
    try {
      const url = getWebSocketURL();
      log('info', `Connecting to: ${url}`);
      
      wsRef.current = new WebSocket(url, WS_CONFIG.protocols);
      
      // Connection timeout
      connectionTimeoutRef.current = setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
          log('error', 'Connection timeout');
          wsRef.current.close();
          setConnectionState(CONNECTION_STATE.ERROR);
          setError('Connection timeout');
        }
      }, WS_CONFIG.connectionTimeout);
      
      wsRef.current.onopen = (event) => {
        if (!mountedRef.current) return;
        
        log('info', 'WebSocket connected');
        setConnectionState(CONNECTION_STATE.CONNECTED);
        setIsConnected(true);
        setError(null);
        setReconnectAttempt(0);
        
        // Clear connection timeout
        if (connectionTimeoutRef.current) {
          clearTimeout(connectionTimeoutRef.current);
          connectionTimeoutRef.current = null;
        }
        
        // Start heartbeat
        startHeartbeat();
        
        // Authenticate if user is logged in
        if (isAuthenticated && user) {
          const authHeaders = getAuthHeaders();
          sendMessage(MESSAGE_TYPES.AUTH, {
            authorization: authHeaders.Authorization,
            userId: user.id
          });
        }
        
        // Re-subscribe to channels
        subscriptions.forEach(channel => {
          sendMessage(MESSAGE_TYPES.SUBSCRIBE, { channel });
        });
        
        // Call connect callback
        if (config.onConnect) {
          config.onConnect(event);
        }
        
        // Track reconnection
        if (reconnectAttempt > 0) {
          log('info', `Reconnected after ${reconnectAttempt} attempts`);
          if (config.onReconnect) {
            config.onReconnect(reconnectAttempt);
          }
        }
      };
      
      wsRef.current.onmessage = handleMessage;
      
      wsRef.current.onclose = (event) => {
        if (!mountedRef.current) return;
        
        log('info', `WebSocket disconnected: ${event.code} - ${event.reason}`);
        setConnectionState(CONNECTION_STATE.DISCONNECTED);
        setIsConnected(false);
        
        // Stop heartbeat
        stopHeartbeat();
        
        // Clear connection timeout
        if (connectionTimeoutRef.current) {
          clearTimeout(connectionTimeoutRef.current);
          connectionTimeoutRef.current = null;
        }
        
        // Call disconnect callback
        if (config.onDisconnect) {
          config.onDisconnect(event);
        }
        
        // Auto-reconnect if enabled and not a normal closure
        if (config.enableAutoReconnect && event.code !== 1000 && event.code !== 1001) {
          scheduleReconnect();
        }
      };
      
      wsRef.current.onerror = (error) => {
        if (!mountedRef.current) return;
        
        log('error', 'WebSocket error', error);
        setConnectionState(CONNECTION_STATE.ERROR);
        setError('Connection error');
        
        // Call error callback
        if (config.onError) {
          config.onError(error);
        }
      };
      
    } catch (error) {
      log('error', 'Failed to create WebSocket connection', error);
      setConnectionState(CONNECTION_STATE.ERROR);
      setError(error.message);
    }
  }, [
    getWebSocketURL, log, startHeartbeat, handleMessage, stopHeartbeat,
    isAuthenticated, user, getAuthHeaders, sendMessage, subscriptions,
    config, reconnectAttempt
  ]);
  
  // Schedule reconnection
  const scheduleReconnect = useCallback(() => {
    if (reconnectAttempt >= WS_CONFIG.maxReconnectAttempts) {
      log('error', 'Max reconnection attempts reached');
      setConnectionState(CONNECTION_STATE.ERROR);
      setError('Failed to reconnect after maximum attempts');
      return;
    }
    
    const delay = WS_CONFIG.reconnectInterval * Math.pow(2, reconnectAttempt);
    log('info', `Scheduling reconnection in ${delay}ms (attempt ${reconnectAttempt + 1})`);
    
    setConnectionState(CONNECTION_STATE.RECONNECTING);
    setReconnectAttempt(prev => prev + 1);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, delay);
  }, [reconnectAttempt, log, connect]);
  
  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    log('info', 'Disconnecting WebSocket');
    
    // Clear timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
    
    // Stop heartbeat
    stopHeartbeat();
    
    // Close connection
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    
    setConnectionState(CONNECTION_STATE.DISCONNECTED);
    setIsConnected(false);
    setError(null);
    setReconnectAttempt(0);
  }, [log, stopHeartbeat]);
  
  // Subscribe to a channel
  const subscribe = useCallback((channel) => {
    setSubscriptions(prev => new Set([...prev, channel]));
    
    if (isConnected) {
      sendMessage(MESSAGE_TYPES.SUBSCRIBE, { channel });
      log('info', `Subscribing to: ${channel}`);
    }
  }, [isConnected, sendMessage, log]);
  
  // Unsubscribe from a channel
  const unsubscribe = useCallback((channel) => {
    setSubscriptions(prev => {
      const newSet = new Set(prev);
      newSet.delete(channel);
      return newSet;
    });
    
    if (isConnected) {
      sendMessage(MESSAGE_TYPES.UNSUBSCRIBE, { channel });
      log('info', `Unsubscribing from: ${channel}`);
    }
  }, [isConnected, sendMessage, log]);
  
  // Register message callback
  const onMessage = useCallback((messageType, callback) => {
    if (!messageCallbacksRef.current.has(messageType)) {
      messageCallbacksRef.current.set(messageType, []);
    }
    
    messageCallbacksRef.current.get(messageType).push(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = messageCallbacksRef.current.get(messageType) || [];
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    };
  }, []);
  
  // Auto-connect on mount if enabled
  useEffect(() => {
    if (config.autoConnect && appState.connectivity.isOnline) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [config.autoConnect, appState.connectivity.isOnline]);
  
  // Reconnect when coming back online
  useEffect(() => {
    if (appState.connectivity.isOnline && connectionState === CONNECTION_STATE.DISCONNECTED) {
      if (config.autoConnect) {
        log('info', 'Reconnecting due to network recovery');
        connect();
      }
    }
  }, [appState.connectivity.isOnline, connectionState, config.autoConnect, connect, log]);
  
  return {
    // State
    connectionState,
    isConnected,
    isConnecting: connectionState === CONNECTION_STATE.CONNECTING,
    isReconnecting: connectionState === CONNECTION_STATE.RECONNECTING,
    error,
    lastMessage,
    messageHistory,
    subscriptions: Array.from(subscriptions),
    reconnectAttempt,
    
    // Actions
    connect,
    disconnect,
    sendMessage,
    subscribe,
    unsubscribe,
    onMessage,
    
    // Utilities
    isOnline: appState.connectivity.isOnline,
    clearError: () => setError(null),
    clearHistory: () => setMessageHistory([]),
    getSubscriptionCount: () => subscriptions.size
  };
};

// Specialized hooks for different use cases
export const useLiveMatches = () => {
  const ws = useWebSocket();
  const [liveMatches, setLiveMatches] = useState([]);
  const [matchUpdates, setMatchUpdates] = useState({});
  
  useEffect(() => {
    if (ws.isConnected) {
      ws.subscribe(CHANNELS.MATCHES_LIVE);
    }
  }, [ws.isConnected, ws.subscribe]);
  
  useEffect(() => {
    const unsubscribeMatchUpdate = ws.onMessage(MESSAGE_TYPES.MATCH_UPDATE, (message) => {
      setMatchUpdates(prev => ({
        ...prev,
        [message.data.matchId]: message.data
      }));
    });
    
    const unsubscribeGoal = ws.onMessage(MESSAGE_TYPES.GOAL_SCORED, (message) => {
      setLiveMatches(prev => prev.map(match => 
        match.id === message.data.matchId 
          ? { ...match, ...message.data }
          : match
      ));
    });
    
    return () => {
      unsubscribeMatchUpdate();
      unsubscribeGoal();
    };
  }, [ws.onMessage]);
  
  return {
    ...ws,
    liveMatches,
    matchUpdates
  };
};

export const useMatchDetail = (matchId) => {
  const ws = useWebSocket();
  const [matchData, setMatchData] = useState(null);
  const [liveStats, setLiveStats] = useState(null);
  
  useEffect(() => {
    if (ws.isConnected && matchId) {
      ws.subscribe(`${CHANNELS.MATCH_DETAIL}${matchId}`);
    }
    
    return () => {
      if (matchId) {
        ws.unsubscribe(`${CHANNELS.MATCH_DETAIL}${matchId}`);
      }
    };
  }, [ws.isConnected, ws.subscribe, ws.unsubscribe, matchId]);
  
  useEffect(() => {
    const unsubscribe = ws.onMessage(MESSAGE_TYPES.LIVE_STATS, (message) => {
      if (message.data.matchId === matchId) {
        setLiveStats(message.data.stats);
      }
    });
    
    return unsubscribe;
  }, [ws.onMessage, matchId]);
  
  return {
    ...ws,
    matchData,
    liveStats
  };
};

export const useNotifications = () => {
  const ws = useWebSocket();
  const { user } = useAuth();
  const [notifications, setNotifications] = useState([]);
  
  useEffect(() => {
    if (ws.isConnected && user) {
      ws.subscribe(`${CHANNELS.USER_NOTIFICATIONS}${user.id}`);
    }
  }, [ws.isConnected, ws.subscribe, user]);
  
  useEffect(() => {
    const unsubscribe = ws.onMessage(MESSAGE_TYPES.USER_NOTIFICATION, (message) => {
      setNotifications(prev => [message.data, ...prev]);
    });
    
    return unsubscribe;
  }, [ws.onMessage]);
  
  const markAsRead = useCallback((notificationId) => {
    ws.sendMessage('mark_notification_read', { notificationId });
    setNotifications(prev => 
      prev.map(n => n.id === notificationId ? { ...n, read: true } : n)
    );
  }, [ws.sendMessage]);
  
  return {
    ...ws,
    notifications,
    markAsRead
  };
};

export const useInjuryUpdates = () => {
  const ws = useWebSocket();
  const [injuryUpdates, setInjuryUpdates] = useState([]);
  
  useEffect(() => {
    if (ws.isConnected) {
      ws.subscribe(CHANNELS.INJURIES);
    }
  }, [ws.isConnected, ws.subscribe]);
  
  useEffect(() => {
    const unsubscribeInjury = ws.onMessage(MESSAGE_TYPES.INJURY_REPORTED, (message) => {
      setInjuryUpdates(prev => [message.data, ...prev]);
    });
    
    const unsubscribeRecovery = ws.onMessage(MESSAGE_TYPES.RECOVERY_UPDATE, (message) => {
      setInjuryUpdates(prev => [message.data, ...prev]);
    });
    
    return () => {
      unsubscribeInjury();
      unsubscribeRecovery();
    };
  }, [ws.onMessage]);
  
  return {
    ...ws,
    injuryUpdates
  };
};

export const usePredictionUpdates = () => {
  const ws = useWebSocket();
  const [predictionUpdates, setPredictionUpdates] = useState({});
  
  useEffect(() => {
    if (ws.isConnected) {
      ws.subscribe(CHANNELS.PREDICTIONS);
    }
  }, [ws.isConnected, ws.subscribe]);
  
  useEffect(() => {
    const unsubscribe = ws.onMessage(MESSAGE_TYPES.PREDICTION_UPDATE, (message) => {
      setPredictionUpdates(prev => ({
        ...prev,
        [message.data.matchId]: message.data
      }));
    });
    
    return unsubscribe;
  }, [ws.onMessage]);
  
  return {
    ...ws,
    predictionUpdates
  };
};

export default useWebSocket;