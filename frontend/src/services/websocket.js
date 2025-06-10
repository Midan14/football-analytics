// WebSocket Service for Football Analytics Platform
// Centralized WebSocket management with auto-reconnection, queuing, and event handling

import authService, { AUTH_EVENTS } from './auth';

// WebSocket configuration
const WS_CONFIG = {
  baseURL: process.env.REACT_APP_WS_URL || 'ws://localhost:3001',
  protocols: ['football-analytics-v1'],
  
  // Connection settings
  connectionTimeout: 10000, // 10 seconds
  maxReconnectAttempts: 10,
  reconnectBaseDelay: 1000, // 1 second
  reconnectMaxDelay: 30000, // 30 seconds
  
  // Heartbeat settings
  heartbeatInterval: 30000, // 30 seconds
  heartbeatTimeout: 5000,   // 5 seconds
  
  // Message settings
  messageQueueSize: 100,
  messageHistorySize: 1000,
  compressionThreshold: 1024, // 1KB
  
  // Retry settings
  maxRetries: 3,
  retryDelay: 1000,
  
  // Debug settings
  enableLogging: process.env.NODE_ENV === 'development',
  enableMessageLogging: false
};

// Connection states
export const CONNECTION_STATES = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting', 
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  CLOSING: 'closing',
  ERROR: 'error'
};

// Message types
export const MESSAGE_TYPES = {
  // System messages
  PING: 'ping',
  PONG: 'pong',
  AUTH: 'auth',
  AUTH_SUCCESS: 'auth_success',
  AUTH_FAILURE: 'auth_failure',
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  SUBSCRIPTION_SUCCESS: 'subscription_success',
  SUBSCRIPTION_ERROR: 'subscription_error',
  ERROR: 'error',
  
  // Match events
  MATCH_START: 'match_start',
  MATCH_END: 'match_end',
  MATCH_UPDATE: 'match_update',
  GOAL_SCORED: 'goal_scored',
  CARD_ISSUED: 'card_issued',
  SUBSTITUTION: 'substitution',
  HALF_TIME: 'half_time',
  FULL_TIME: 'full_time',
  LIVE_STATS: 'live_stats',
  LIVE_COMMENTARY: 'live_commentary',
  
  // Prediction events
  PREDICTION_UPDATE: 'prediction_update',
  PREDICTION_RESULT: 'prediction_result',
  MODEL_UPDATE: 'model_update',
  CONFIDENCE_CHANGE: 'confidence_change',
  
  // Injury events
  INJURY_REPORTED: 'injury_reported',
  INJURY_UPDATE: 'injury_update',
  RECOVERY_UPDATE: 'recovery_update',
  MEDICAL_CLEARANCE: 'medical_clearance',
  
  // User events
  USER_NOTIFICATION: 'user_notification',
  FAVORITE_UPDATE: 'favorite_update',
  PREFERENCE_SYNC: 'preference_sync',
  
  // System events
  SYSTEM_MAINTENANCE: 'system_maintenance',
  FEATURE_UPDATE: 'feature_update',
  BROADCAST_MESSAGE: 'broadcast_message'
};

// Subscription channels
export const CHANNELS = {
  // Global channels
  SYSTEM: 'system',
  BROADCASTS: 'broadcasts',
  
  // Match channels
  MATCHES_LIVE: 'matches:live',
  MATCHES_ALL: 'matches:all',
  MATCH_DETAIL: 'match:detail:',      // + matchId
  MATCH_COMMENTARY: 'match:commentary:', // + matchId
  
  // Team channels
  TEAM_UPDATES: 'team:updates:',      // + teamId
  TEAM_MATCHES: 'team:matches:',      // + teamId
  TEAM_NEWS: 'team:news:',           // + teamId
  
  // Player channels  
  PLAYER_UPDATES: 'player:updates:',  // + playerId
  PLAYER_STATS: 'player:stats:',     // + playerId
  
  // League channels
  LEAGUE_UPDATES: 'league:updates:',  // + leagueId
  LEAGUE_STANDINGS: 'league:standings:', // + leagueId
  LEAGUE_MATCHES: 'league:matches:',  // + leagueId
  
  // Prediction channels
  PREDICTIONS: 'predictions',
  PREDICTIONS_MATCH: 'predictions:match:', // + matchId
  PREDICTIONS_LEAGUE: 'predictions:league:', // + leagueId
  
  // Injury channels
  INJURIES: 'injuries',
  INJURIES_TEAM: 'injuries:team:',    // + teamId
  INJURIES_PLAYER: 'injuries:player:', // + playerId
  
  // User channels
  USER_NOTIFICATIONS: 'user:notifications:', // + userId
  USER_FAVORITES: 'user:favorites:',  // + userId
  USER_PREFERENCES: 'user:preferences:', // + userId
  
  // Geographic channels
  COUNTRY_UPDATES: 'country:updates:', // + countryCode
  CONFEDERATION_UPDATES: 'confederation:updates:' // + confederation
};

// Event priorities
const EVENT_PRIORITIES = {
  CRITICAL: 1,    // System errors, security issues
  HIGH: 2,        // Goals, red cards, match end
  MEDIUM: 3,      // Yellow cards, substitutions, injuries
  LOW: 4,         // Stats updates, commentary
  BACKGROUND: 5   // Heartbeat, subscriptions
};

// WebSocket Service Class
class WebSocketService {
  constructor() {
    this.ws = null;
    this.connectionState = CONNECTION_STATES.DISCONNECTED;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.heartbeatTimeoutTimer = null;
    this.lastHeartbeatResponse = null;
    
    // Message management
    this.messageQueue = [];
    this.messageHistory = [];
    this.pendingMessages = new Map();
    this.messageIdCounter = 0;
    
    // Subscription management
    this.subscriptions = new Set();
    this.subscriptionCallbacks = new Map();
    this.globalEventListeners = new Map();
    
    // Authentication
    this.isAuthenticated = false;
    this.authToken = null;
    this.userId = null;
    
    // Connection management
    this.connectionPromise = null;
    this.isDestroyed = false;
    this.connectionId = null;
    
    // Performance monitoring
    this.stats = {
      messagesReceived: 0,
      messagesSent: 0,
      reconnections: 0,
      errors: 0,
      avgLatency: 0,
      lastConnected: null,
      totalUptime: 0
    };
    
    // Initialize
    this.initializeAuthListeners();
    this.initializeNetworkListeners();
  }

  // Initialize authentication listeners
  initializeAuthListeners() {
    authService.on(AUTH_EVENTS.LOGIN_SUCCESS, (data) => {
      this.handleAuthSuccess(data.user);
    });
    
    authService.on(AUTH_EVENTS.LOGOUT, () => {
      this.handleAuthLogout();
    });
    
    authService.on(AUTH_EVENTS.TOKEN_REFRESHED, (data) => {
      this.handleTokenRefresh(data);
    });
  }

  // Initialize network listeners
  initializeNetworkListeners() {
    window.addEventListener('online', () => {
      this.log('info', 'Network online - attempting reconnection');
      if (this.connectionState === CONNECTION_STATES.DISCONNECTED) {
        this.connect();
      }
    });
    
    window.addEventListener('offline', () => {
      this.log('warn', 'Network offline - connection will be restored when online');
    });
    
    window.addEventListener('beforeunload', () => {
      this.disconnect(false); // Don't attempt reconnection
    });
  }

  // Logging utility
  log(level, message, data = null) {
    if (!WS_CONFIG.enableLogging) return;
    
    const timestamp = new Date().toISOString();
    const logMessage = `[WebSocket] ${timestamp} - ${message}`;
    
    console[level](logMessage, data || '');
    
    // Send to analytics in production
    if (level === 'error' && process.env.NODE_ENV === 'production') {
      this.trackError(message, data);
    }
  }

  // Analytics tracking
  trackError(message, data) {
    try {
      if (window.gtag) {
        window.gtag('event', 'websocket_error', {
          event_category: 'WebSocket',
          event_label: message,
          value: 1,
          custom_map: { error_data: JSON.stringify(data) }
        });
      }
    } catch (error) {
      console.error('Failed to track WebSocket error:', error);
    }
  }

  // Connection management
  async connect(force = false) {
    if (this.isDestroyed) return false;
    
    if (this.connectionState === CONNECTION_STATES.CONNECTED && !force) {
      this.log('warn', 'Already connected');
      return true;
    }
    
    if (this.connectionState === CONNECTION_STATES.CONNECTING && !force) {
      this.log('warn', 'Connection already in progress');
      return this.connectionPromise;
    }
    
    this.connectionState = CONNECTION_STATES.CONNECTING;
    this.connectionPromise = this.establishConnection();
    
    try {
      const result = await this.connectionPromise;
      return result;
    } catch (error) {
      this.log('error', 'Connection failed', error);
      return false;
    } finally {
      this.connectionPromise = null;
    }
  }

  // Establish WebSocket connection
  async establishConnection() {
    return new Promise((resolve, reject) => {
      try {
        const url = this.buildWebSocketURL();
        this.log('info', `Connecting to: ${url}`);
        
        this.ws = new WebSocket(url, WS_CONFIG.protocols);
        
        const connectionTimeout = setTimeout(() => {
          this.log('error', 'Connection timeout');
          this.ws.close();
          reject(new Error('Connection timeout'));
        }, WS_CONFIG.connectionTimeout);

        this.ws.onopen = (event) => {
          clearTimeout(connectionTimeout);
          this.handleConnectionOpen(event);
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          this.handleConnectionClose(event);
          if (this.connectionState === CONNECTION_STATES.CONNECTING) {
            reject(new Error(`Connection failed: ${event.code} ${event.reason}`));
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          this.handleConnectionError(error);
          if (this.connectionState === CONNECTION_STATES.CONNECTING) {
            reject(error);
          }
        };
        
      } catch (error) {
        this.log('error', 'Failed to create WebSocket connection', error);
        reject(error);
      }
    });
  }

  // Build WebSocket URL with authentication
  buildWebSocketURL() {
    let url = WS_CONFIG.baseURL;
    
    // Add authentication parameters
    if (authService.isAuthenticated()) {
      const user = authService.getUser();
      const token = authService.getToken();
      
      const params = new URLSearchParams({
        token: token || '',
        userId: user?.id || '',
        sessionId: authService.getSession()?.id || '',
        timestamp: Date.now().toString()
      });
      
      url += `?${params.toString()}`;
    }
    
    return url;
  }

  // Handle connection open
  handleConnectionOpen(event) {
    this.log('info', 'WebSocket connection established');
    
    this.connectionState = CONNECTION_STATES.CONNECTED;
    this.reconnectAttempts = 0;
    this.stats.lastConnected = new Date();
    this.stats.reconnections += (this.stats.lastConnected ? 1 : 0);
    
    // Generate connection ID
    this.connectionId = `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Authenticate if user is logged in
    if (authService.isAuthenticated()) {
      this.authenticate();
    }
    
    // Process message queue
    this.processMessageQueue();
    
    // Re-establish subscriptions
    this.reestablishSubscriptions();
    
    // Emit connection event
    this.emit('connected', { connectionId: this.connectionId, reconnect: this.stats.reconnections > 0 });
  }

  // Handle connection close
  handleConnectionClose(event) {
    this.log('info', `WebSocket connection closed: ${event.code} - ${event.reason}`);
    
    const wasConnected = this.connectionState === CONNECTION_STATES.CONNECTED;
    this.connectionState = CONNECTION_STATES.DISCONNECTED;
    this.connectionId = null;
    
    // Stop heartbeat
    this.stopHeartbeat();
    
    // Clear authentication
    this.isAuthenticated = false;
    
    // Emit disconnection event
    this.emit('disconnected', { code: event.code, reason: event.reason, wasConnected });
    
    // Attempt reconnection if not a clean close
    if (!this.isDestroyed && wasConnected && event.code !== 1000 && event.code !== 1001) {
      this.scheduleReconnection();
    }
  }

  // Handle connection error
  handleConnectionError(error) {
    this.log('error', 'WebSocket connection error', error);
    
    this.connectionState = CONNECTION_STATES.ERROR;
    this.stats.errors++;
    
    this.emit('error', error);
  }

  // Schedule reconnection with exponential backoff
  scheduleReconnection() {
    if (this.isDestroyed || this.reconnectAttempts >= WS_CONFIG.maxReconnectAttempts) {
      this.log('error', 'Max reconnection attempts reached');
      this.emit('max_reconnect_attempts');
      return;
    }
    
    this.connectionState = CONNECTION_STATES.RECONNECTING;
    this.reconnectAttempts++;
    
    // Calculate delay with exponential backoff and jitter
    const baseDelay = WS_CONFIG.reconnectBaseDelay;
    const exponentialDelay = Math.min(
      baseDelay * Math.pow(2, this.reconnectAttempts - 1),
      WS_CONFIG.reconnectMaxDelay
    );
    const jitter = exponentialDelay * 0.1 * Math.random();
    const delay = exponentialDelay + jitter;
    
    this.log('info', `Scheduling reconnection attempt ${this.reconnectAttempts} in ${Math.round(delay)}ms`);
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
    
    this.emit('reconnecting', { attempt: this.reconnectAttempts, delay });
  }

  // Disconnect
  disconnect(allowReconnect = false) {
    this.log('info', `Disconnecting WebSocket${allowReconnect ? ' (allowing reconnect)' : ''}`);
    
    if (!allowReconnect) {
      this.isDestroyed = true;
    }
    
    // Clear timers
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.stopHeartbeat();
    
    // Close connection
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.connectionState = CONNECTION_STATES.CLOSING;
      this.ws.close(1000, 'Client disconnect');
    } else {
      this.connectionState = CONNECTION_STATES.DISCONNECTED;
    }
    
    this.ws = null;
    this.connectionId = null;
    this.isAuthenticated = false;
  }

  // Heartbeat management
  startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
    }, WS_CONFIG.heartbeatInterval);
    
    this.log('debug', 'Heartbeat started');
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
    
    this.log('debug', 'Heartbeat stopped');
  }

  sendHeartbeat() {
    if (this.connectionState !== CONNECTION_STATES.CONNECTED) return;
    
    const heartbeatData = {
      timestamp: Date.now(),
      connectionId: this.connectionId,
      stats: this.getConnectionStats()
    };
    
    this.sendMessage(MESSAGE_TYPES.PING, heartbeatData);
    
    // Set timeout for pong response
    this.heartbeatTimeoutTimer = setTimeout(() => {
      this.log('warn', 'Heartbeat timeout - connection may be lost');
      this.handleHeartbeatTimeout();
    }, WS_CONFIG.heartbeatTimeout);
  }

  handleHeartbeatTimeout() {
    this.log('error', 'Heartbeat timeout - closing connection');
    if (this.ws) {
      this.ws.close(1006, 'Heartbeat timeout');
    }
  }

  handleHeartbeatResponse(data) {
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
    
    this.lastHeartbeatResponse = Date.now();
    
    // Calculate latency
    if (data.timestamp) {
      const latency = Date.now() - data.timestamp;
      this.updateLatencyStats(latency);
    }
    
    this.log('debug', 'Heartbeat response received');
  }

  // Authentication
  authenticate() {
    if (!authService.isAuthenticated()) {
      this.log('warn', 'Cannot authenticate - user not logged in');
      return;
    }
    
    const user = authService.getUser();
    const token = authService.getToken();
    
    const authData = {
      token,
      userId: user.id,
      sessionId: authService.getSession()?.id,
      timestamp: Date.now()
    };
    
    this.sendMessage(MESSAGE_TYPES.AUTH, authData);
    this.log('info', 'Authentication message sent');
  }

  handleAuthSuccess(data) {
    this.log('info', 'WebSocket authentication successful');
    this.isAuthenticated = true;
    this.userId = data.userId;
    
    // Subscribe to user-specific channels
    this.subscribeToUserChannels();
    
    this.emit('authenticated', data);
  }

  handleAuthFailure(data) {
    this.log('error', 'WebSocket authentication failed', data);
    this.isAuthenticated = false;
    this.userId = null;
    
    this.emit('auth_failed', data);
    
    // Disconnect and retry authentication
    this.disconnect();
    setTimeout(() => this.connect(), 5000);
  }

  handleAuthSuccess(user) {
    this.authToken = authService.getToken();
    this.userId = user.id;
    
    // Reconnect with authentication
    if (this.connectionState === CONNECTION_STATES.CONNECTED) {
      this.authenticate();
    } else if (this.connectionState === CONNECTION_STATES.DISCONNECTED) {
      this.connect();
    }
  }

  handleAuthLogout() {
    this.authToken = null;
    this.userId = null;
    this.isAuthenticated = false;
    
    // Clear user-specific subscriptions
    this.clearUserSubscriptions();
    
    // Don't disconnect, but stop user-specific functionality
    this.log('info', 'User logged out - cleared authentication state');
  }

  handleTokenRefresh(data) {
    this.authToken = data.accessToken;
    
    // Re-authenticate with new token if connected
    if (this.connectionState === CONNECTION_STATES.CONNECTED && this.isAuthenticated) {
      this.authenticate();
    }
  }

  // Message handling
  handleMessage(event) {
    try {
      const message = JSON.parse(event.data);
      this.stats.messagesReceived++;
      
      if (WS_CONFIG.enableMessageLogging) {
        this.log('debug', `Received message: ${message.type}`, message);
      }
      
      // Add to message history
      this.addToMessageHistory(message);
      
      // Handle system messages
      this.handleSystemMessage(message);
      
      // Emit to listeners
      this.emitMessage(message);
      
    } catch (error) {
      this.log('error', 'Failed to parse message', { error, data: event.data });
    }
  }

  handleSystemMessage(message) {
    switch (message.type) {
      case MESSAGE_TYPES.PONG:
        this.handleHeartbeatResponse(message.data);
        break;
        
      case MESSAGE_TYPES.AUTH_SUCCESS:
        this.handleAuthSuccess(message.data);
        break;
        
      case MESSAGE_TYPES.AUTH_FAILURE:
        this.handleAuthFailure(message.data);
        break;
        
      case MESSAGE_TYPES.SUBSCRIPTION_SUCCESS:
        this.handleSubscriptionSuccess(message.data);
        break;
        
      case MESSAGE_TYPES.SUBSCRIPTION_ERROR:
        this.handleSubscriptionError(message.data);
        break;
        
      case MESSAGE_TYPES.ERROR:
        this.handleServerError(message.data);
        break;
        
      case MESSAGE_TYPES.SYSTEM_MAINTENANCE:
        this.handleSystemMaintenance(message.data);
        break;
        
      default:
        // Handle application-specific messages
        this.handleApplicationMessage(message);
    }
  }

  handleApplicationMessage(message) {
    // Emit events for different message types
    switch (message.type) {
      case MESSAGE_TYPES.GOAL_SCORED:
        this.emit('goal_scored', message.data);
        this.emit('match_event', { type: 'goal', ...message.data });
        break;
        
      case MESSAGE_TYPES.CARD_ISSUED:
        this.emit('card_issued', message.data);
        this.emit('match_event', { type: 'card', ...message.data });
        break;
        
      case MESSAGE_TYPES.SUBSTITUTION:
        this.emit('substitution', message.data);
        this.emit('match_event', { type: 'substitution', ...message.data });
        break;
        
      case MESSAGE_TYPES.MATCH_START:
        this.emit('match_start', message.data);
        break;
        
      case MESSAGE_TYPES.MATCH_END:
        this.emit('match_end', message.data);
        break;
        
      case MESSAGE_TYPES.INJURY_REPORTED:
        this.emit('injury_reported', message.data);
        break;
        
      case MESSAGE_TYPES.PREDICTION_UPDATE:
        this.emit('prediction_update', message.data);
        break;
        
      case MESSAGE_TYPES.USER_NOTIFICATION:
        this.emit('user_notification', message.data);
        break;
        
      default:
        this.emit('message', message);
    }
  }

  // Message sending
  sendMessage(type, data = {}, options = {}) {
    const message = {
      id: this.generateMessageId(),
      type,
      data,
      timestamp: new Date().toISOString(),
      connectionId: this.connectionId,
      userId: this.userId,
      ...options
    };
    
    if (this.connectionState === CONNECTION_STATES.CONNECTED && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
        this.stats.messagesSent++;
        
        if (WS_CONFIG.enableMessageLogging) {
          this.log('debug', `Sent message: ${type}`, message);
        }
        
        return true;
      } catch (error) {
        this.log('error', 'Failed to send message', { error, message });
        this.queueMessage(message);
        return false;
      }
    } else {
      this.log('warn', `Cannot send message - not connected (state: ${this.connectionState})`);
      this.queueMessage(message);
      return false;
    }
  }

  // Message queue management
  queueMessage(message) {
    if (this.messageQueue.length >= WS_CONFIG.messageQueueSize) {
      // Remove oldest message
      this.messageQueue.shift();
      this.log('warn', 'Message queue full - dropped oldest message');
    }
    
    this.messageQueue.push(message);
    this.log('debug', `Message queued: ${message.type} (queue size: ${this.messageQueue.length})`);
  }

  processMessageQueue() {
    if (this.messageQueue.length === 0) return;
    
    this.log('info', `Processing message queue (${this.messageQueue.length} messages)`);
    
    const messages = [...this.messageQueue];
    this.messageQueue = [];
    
    messages.forEach(message => {
      // Update timestamp
      message.timestamp = new Date().toISOString();
      message.connectionId = this.connectionId;
      
      try {
        this.ws.send(JSON.stringify(message));
        this.stats.messagesSent++;
      } catch (error) {
        this.log('error', 'Failed to send queued message', { error, message });
        // Re-queue if failed
        this.queueMessage(message);
      }
    });
  }

  // Subscription management
  subscribe(channel, callback = null) {
    this.subscriptions.add(channel);
    
    if (callback) {
      if (!this.subscriptionCallbacks.has(channel)) {
        this.subscriptionCallbacks.set(channel, new Set());
      }
      this.subscriptionCallbacks.get(channel).add(callback);
    }
    
    if (this.connectionState === CONNECTION_STATES.CONNECTED) {
      this.sendMessage(MESSAGE_TYPES.SUBSCRIBE, { channel });
    }
    
    this.log('info', `Subscribed to channel: ${channel}`);
    
    // Return unsubscribe function
    return () => this.unsubscribe(channel, callback);
  }

  unsubscribe(channel, callback = null) {
    if (callback && this.subscriptionCallbacks.has(channel)) {
      this.subscriptionCallbacks.get(channel).delete(callback);
      
      // If no more callbacks for this channel, unsubscribe completely
      if (this.subscriptionCallbacks.get(channel).size === 0) {
        this.subscriptionCallbacks.delete(channel);
        this.subscriptions.delete(channel);
        
        if (this.connectionState === CONNECTION_STATES.CONNECTED) {
          this.sendMessage(MESSAGE_TYPES.UNSUBSCRIBE, { channel });
        }
      }
    } else {
      // Remove channel completely
      this.subscriptions.delete(channel);
      this.subscriptionCallbacks.delete(channel);
      
      if (this.connectionState === CONNECTION_STATES.CONNECTED) {
        this.sendMessage(MESSAGE_TYPES.UNSUBSCRIBE, { channel });
      }
    }
    
    this.log('info', `Unsubscribed from channel: ${channel}`);
  }

  reestablishSubscriptions() {
    if (this.subscriptions.size === 0) return;
    
    this.log('info', `Re-establishing ${this.subscriptions.size} subscriptions`);
    
    this.subscriptions.forEach(channel => {
      this.sendMessage(MESSAGE_TYPES.SUBSCRIBE, { channel });
    });
  }

  subscribeToUserChannels() {
    if (!this.userId) return;
    
    // Subscribe to user-specific channels
    this.subscribe(`${CHANNELS.USER_NOTIFICATIONS}${this.userId}`);
    this.subscribe(`${CHANNELS.USER_FAVORITES}${this.userId}`);
    this.subscribe(`${CHANNELS.USER_PREFERENCES}${this.userId}`);
  }

  clearUserSubscriptions() {
    if (!this.userId) return;
    
    // Unsubscribe from user-specific channels
    this.unsubscribe(`${CHANNELS.USER_NOTIFICATIONS}${this.userId}`);
    this.unsubscribe(`${CHANNELS.USER_FAVORITES}${this.userId}`);
    this.unsubscribe(`${CHANNELS.USER_PREFERENCES}${this.userId}`);
  }

  handleSubscriptionSuccess(data) {
    this.log('info', `Subscription successful: ${data.channel}`);
    this.emit('subscription_success', data);
  }

  handleSubscriptionError(data) {
    this.log('error', `Subscription failed: ${data.channel}`, data.error);
    this.emit('subscription_error', data);
  }

  // Event emission system
  on(event, callback) {
    if (!this.globalEventListeners.has(event)) {
      this.globalEventListeners.set(event, new Set());
    }
    
    this.globalEventListeners.get(event).add(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.globalEventListeners.get(event);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.globalEventListeners.delete(event);
        }
      }
    };
  }

  off(event, callback) {
    const listeners = this.globalEventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
      if (listeners.size === 0) {
        this.globalEventListeners.delete(event);
      }
    }
  }

  emit(event, data = null) {
    const listeners = this.globalEventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          this.log('error', 'Event callback error', { event, error });
        }
      });
    }
  }

  emitMessage(message) {
    // Emit to global message listeners
    this.emit('message', message);
    
    // Emit to type-specific listeners
    this.emit(message.type, message.data);
    
    // Emit to channel-specific listeners
    if (message.channel) {
      const callbacks = this.subscriptionCallbacks.get(message.channel);
      if (callbacks) {
        callbacks.forEach(callback => {
          try {
            callback(message);
          } catch (error) {
            this.log('error', 'Channel callback error', { channel: message.channel, error });
          }
        });
      }
    }
  }

  // System event handlers
  handleServerError(data) {
    this.log('error', 'Server error received', data);
    this.emit('server_error', data);
  }

  handleSystemMaintenance(data) {
    this.log('warn', 'System maintenance notification', data);
    this.emit('system_maintenance', data);
  }

  // Utility methods
  generateMessageId() {
    return `msg_${Date.now()}_${++this.messageIdCounter}`;
  }

  addToMessageHistory(message) {
    this.messageHistory.unshift(message);
    
    if (this.messageHistory.length > WS_CONFIG.messageHistorySize) {
      this.messageHistory = this.messageHistory.slice(0, WS_CONFIG.messageHistorySize);
    }
  }

  updateLatencyStats(latency) {
    if (this.stats.avgLatency === 0) {
      this.stats.avgLatency = latency;
    } else {
      // Calculate moving average
      this.stats.avgLatency = (this.stats.avgLatency * 0.9) + (latency * 0.1);
    }
  }

  getConnectionStats() {
    return {
      state: this.connectionState,
      connected: this.connectionState === CONNECTION_STATES.CONNECTED,
      authenticated: this.isAuthenticated,
      subscriptions: this.subscriptions.size,
      messageQueue: this.messageQueue.length,
      ...this.stats
    };
  }

  getMessageHistory(type = null, limit = 50) {
    let history = this.messageHistory;
    
    if (type) {
      history = history.filter(msg => msg.type === type);
    }
    
    return history.slice(0, limit);
  }

  // High-level convenience methods
  subscribeToLiveMatches(callback) {
    return this.subscribe(CHANNELS.MATCHES_LIVE, callback);
  }

  subscribeToMatch(matchId, callback) {
    return this.subscribe(`${CHANNELS.MATCH_DETAIL}${matchId}`, callback);
  }

  subscribeToTeam(teamId, callback) {
    return this.subscribe(`${CHANNELS.TEAM_UPDATES}${teamId}`, callback);
  }

  subscribeToPlayer(playerId, callback) {
    return this.subscribe(`${CHANNELS.PLAYER_UPDATES}${playerId}`, callback);
  }

  subscribeToInjuries(callback) {
    return this.subscribe(CHANNELS.INJURIES, callback);
  }

  subscribeToPredictions(callback) {
    return this.subscribe(CHANNELS.PREDICTIONS, callback);
  }

  // Cleanup
  destroy() {
    this.log('info', 'Destroying WebSocket service');
    
    this.isDestroyed = true;
    this.disconnect(false);
    
    // Clear all timers
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    // Clear all listeners
    this.globalEventListeners.clear();
    this.subscriptionCallbacks.clear();
    this.subscriptions.clear();
    
    // Clear message queue and history
    this.messageQueue = [];
    this.messageHistory = [];
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// Export the service and utilities
export default webSocketService;

export {
  CHANNELS,
  CONNECTION_STATES,
  EVENT_PRIORITIES, MESSAGE_TYPES, WebSocketService
};
