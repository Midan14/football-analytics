import { createContext, useCallback, useContext, useEffect, useReducer } from 'react';
import { useApp } from './AppContext';

// Initial authentication state
const initialAuthState = {
  // User Data
  user: null,
  profile: null,
  
  // Authentication Status
  isAuthenticated: false,
  isLoading: false,
  isInitialized: false,
  
  // Tokens
  accessToken: null,
  refreshToken: null,
  tokenExpiry: null,
  
  // Session Management
  sessionId: null,
  lastActivity: null,
  sessionTimeout: 30 * 60 * 1000, // 30 minutes
  
  // User Preferences
  preferences: {
    favoriteTeams: [],
    favoritePlayers: [],
    favoriteLeagues: [],
    defaultView: 'cards',
    notifications: {
      matchResults: true,
      injuryUpdates: true,
      predictionAlerts: true,
      favoriteTeamNews: true
    },
    privacy: {
      profileVisibility: 'public',
      shareStatistics: true,
      allowAnalytics: true
    }
  },
  
  // Permissions & Roles
  roles: [],
  permissions: [],
  subscription: {
    type: 'free', // 'free' | 'premium' | 'pro'
    features: [],
    expiresAt: null
  },
  
  // Error Handling
  error: null,
  loginAttempts: 0,
  lockedUntil: null,
  
  // Device & Security
  devices: [],
  currentDevice: null,
  twoFactorEnabled: false,
  twoFactorVerified: false,
  
  // Analytics
  analytics: {
    loginCount: 0,
    lastLogin: null,
    totalSessionTime: 0,
    featuresUsed: {},
    favoritesCount: 0
  }
};

// Auth Action Types
export const AuthActionTypes = {
  // Authentication
  AUTH_START: 'AUTH_START',
  AUTH_SUCCESS: 'AUTH_SUCCESS',
  AUTH_FAILURE: 'AUTH_FAILURE',
  AUTH_LOGOUT: 'AUTH_LOGOUT',
  AUTH_INITIALIZED: 'AUTH_INITIALIZED',
  
  // User Profile
  SET_USER: 'SET_USER',
  UPDATE_PROFILE: 'UPDATE_PROFILE',
  SET_PREFERENCES: 'SET_PREFERENCES',
  UPDATE_PREFERENCE: 'UPDATE_PREFERENCE',
  
  // Tokens
  SET_TOKENS: 'SET_TOKENS',
  REFRESH_TOKEN_SUCCESS: 'REFRESH_TOKEN_SUCCESS',
  REFRESH_TOKEN_FAILURE: 'REFRESH_TOKEN_FAILURE',
  CLEAR_TOKENS: 'CLEAR_TOKENS',
  
  // Session
  UPDATE_ACTIVITY: 'UPDATE_ACTIVITY',
  SESSION_TIMEOUT: 'SESSION_TIMEOUT',
  SET_SESSION: 'SET_SESSION',
  
  // Security
  SET_TWO_FACTOR: 'SET_TWO_FACTOR',
  VERIFY_TWO_FACTOR: 'VERIFY_TWO_FACTOR',
  ADD_DEVICE: 'ADD_DEVICE',
  REMOVE_DEVICE: 'REMOVE_DEVICE',
  
  // Subscription
  SET_SUBSCRIPTION: 'SET_SUBSCRIPTION',
  UPDATE_SUBSCRIPTION: 'UPDATE_SUBSCRIPTION',
  
  // Error Handling
  SET_AUTH_ERROR: 'SET_AUTH_ERROR',
  CLEAR_AUTH_ERROR: 'CLEAR_AUTH_ERROR',
  INCREMENT_LOGIN_ATTEMPTS: 'INCREMENT_LOGIN_ATTEMPTS',
  RESET_LOGIN_ATTEMPTS: 'RESET_LOGIN_ATTEMPTS',
  LOCK_ACCOUNT: 'LOCK_ACCOUNT',
  
  // Analytics
  UPDATE_ANALYTICS: 'UPDATE_ANALYTICS',
  TRACK_FEATURE_USAGE: 'TRACK_FEATURE_USAGE'
};

// Auth Reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case AuthActionTypes.AUTH_START:
      return {
        ...state,
        isLoading: true,
        error: null
      };
      
    case AuthActionTypes.AUTH_SUCCESS:
      return {
        ...state,
        isLoading: false,
        isAuthenticated: true,
        user: action.payload.user,
        accessToken: action.payload.accessToken,
        refreshToken: action.payload.refreshToken,
        tokenExpiry: action.payload.tokenExpiry,
        sessionId: action.payload.sessionId,
        lastActivity: new Date(),
        loginAttempts: 0,
        lockedUntil: null,
        error: null,
        analytics: {
          ...state.analytics,
          loginCount: state.analytics.loginCount + 1,
          lastLogin: new Date()
        }
      };
      
    case AuthActionTypes.AUTH_FAILURE:
      return {
        ...state,
        isLoading: false,
        isAuthenticated: false,
        error: action.payload,
        loginAttempts: state.loginAttempts + 1
      };
      
    case AuthActionTypes.AUTH_LOGOUT:
      return {
        ...initialAuthState,
        isInitialized: true,
        analytics: {
          ...initialAuthState.analytics,
          totalSessionTime: state.analytics.totalSessionTime + (new Date() - new Date(state.lastActivity))
        }
      };
      
    case AuthActionTypes.AUTH_INITIALIZED:
      return {
        ...state,
        isInitialized: true
      };
      
    case AuthActionTypes.SET_USER:
      return {
        ...state,
        user: action.payload
      };
      
    case AuthActionTypes.UPDATE_PROFILE:
      return {
        ...state,
        profile: {
          ...state.profile,
          ...action.payload
        }
      };
      
    case AuthActionTypes.SET_PREFERENCES:
      return {
        ...state,
        preferences: action.payload
      };
      
    case AuthActionTypes.UPDATE_PREFERENCE:
      return {
        ...state,
        preferences: {
          ...state.preferences,
          [action.payload.key]: action.payload.value
        }
      };
      
    case AuthActionTypes.SET_TOKENS:
      return {
        ...state,
        accessToken: action.payload.accessToken,
        refreshToken: action.payload.refreshToken,
        tokenExpiry: action.payload.tokenExpiry
      };
      
    case AuthActionTypes.REFRESH_TOKEN_SUCCESS:
      return {
        ...state,
        accessToken: action.payload.accessToken,
        tokenExpiry: action.payload.tokenExpiry,
        lastActivity: new Date()
      };
      
    case AuthActionTypes.REFRESH_TOKEN_FAILURE:
      return {
        ...initialAuthState,
        isInitialized: true,
        error: 'Session expired. Please login again.'
      };
      
    case AuthActionTypes.CLEAR_TOKENS:
      return {
        ...state,
        accessToken: null,
        refreshToken: null,
        tokenExpiry: null
      };
      
    case AuthActionTypes.UPDATE_ACTIVITY:
      return {
        ...state,
        lastActivity: new Date()
      };
      
    case AuthActionTypes.SESSION_TIMEOUT:
      return {
        ...initialAuthState,
        isInitialized: true,
        error: 'Session timed out. Please login again.'
      };
      
    case AuthActionTypes.SET_SESSION:
      return {
        ...state,
        sessionId: action.payload.sessionId,
        lastActivity: new Date()
      };
      
    case AuthActionTypes.SET_TWO_FACTOR:
      return {
        ...state,
        twoFactorEnabled: action.payload
      };
      
    case AuthActionTypes.VERIFY_TWO_FACTOR:
      return {
        ...state,
        twoFactorVerified: action.payload
      };
      
    case AuthActionTypes.ADD_DEVICE:
      return {
        ...state,
        devices: [...state.devices, action.payload],
        currentDevice: action.payload
      };
      
    case AuthActionTypes.REMOVE_DEVICE:
      return {
        ...state,
        devices: state.devices.filter(device => device.id !== action.payload)
      };
      
    case AuthActionTypes.SET_SUBSCRIPTION:
      return {
        ...state,
        subscription: action.payload
      };
      
    case AuthActionTypes.UPDATE_SUBSCRIPTION:
      return {
        ...state,
        subscription: {
          ...state.subscription,
          ...action.payload
        }
      };
      
    case AuthActionTypes.SET_AUTH_ERROR:
      return {
        ...state,
        error: action.payload
      };
      
    case AuthActionTypes.CLEAR_AUTH_ERROR:
      return {
        ...state,
        error: null
      };
      
    case AuthActionTypes.INCREMENT_LOGIN_ATTEMPTS:
      return {
        ...state,
        loginAttempts: state.loginAttempts + 1
      };
      
    case AuthActionTypes.RESET_LOGIN_ATTEMPTS:
      return {
        ...state,
        loginAttempts: 0
      };
      
    case AuthActionTypes.LOCK_ACCOUNT:
      return {
        ...state,
        lockedUntil: action.payload
      };
      
    case AuthActionTypes.UPDATE_ANALYTICS:
      return {
        ...state,
        analytics: {
          ...state.analytics,
          ...action.payload
        }
      };
      
    case AuthActionTypes.TRACK_FEATURE_USAGE:
      return {
        ...state,
        analytics: {
          ...state.analytics,
          featuresUsed: {
            ...state.analytics.featuresUsed,
            [action.payload]: (state.analytics.featuresUsed[action.payload] || 0) + 1
          }
        }
      };
      
    default:
      return state;
  }
};

// Create Auth Context
const AuthContext = createContext();

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [authState, dispatch] = useReducer(authReducer, initialAuthState);
  const { actions: appActions } = useApp();

  // API Base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

  // Initialize authentication on mount
  useEffect(() => {
    initializeAuth();
  }, []);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!authState.accessToken || !authState.tokenExpiry) return;

    const timeUntilExpiry = new Date(authState.tokenExpiry) - new Date();
    const refreshTime = Math.max(timeUntilExpiry - 5 * 60 * 1000, 30 * 1000); // 5 minutes before expiry

    const refreshTimer = setTimeout(() => {
      refreshAccessToken();
    }, refreshTime);

    return () => clearTimeout(refreshTimer);
  }, [authState.accessToken, authState.tokenExpiry]);

  // Session timeout monitoring
  useEffect(() => {
    if (!authState.isAuthenticated || !authState.lastActivity) return;

    const checkSessionTimeout = () => {
      const timeSinceActivity = new Date() - new Date(authState.lastActivity);
      if (timeSinceActivity > authState.sessionTimeout) {
        handleSessionTimeout();
      }
    };

    const sessionTimer = setInterval(checkSessionTimeout, 60 * 1000); // Check every minute

    return () => clearInterval(sessionTimer);
  }, [authState.isAuthenticated, authState.lastActivity, authState.sessionTimeout]);

  // Initialize authentication from stored data
  const initializeAuth = useCallback(async () => {
    try {
      const storedToken = localStorage.getItem('football-analytics-token');
      const storedRefreshToken = localStorage.getItem('football-analytics-refresh-token');
      const storedUser = localStorage.getItem('football-analytics-user');

      if (storedToken && storedRefreshToken && storedUser) {
        const user = JSON.parse(storedUser);
        
        // Verify token validity
        const response = await fetch(`${API_BASE_URL}/auth/verify`, {
          headers: {
            'Authorization': `Bearer ${storedToken}`
          }
        });

        if (response.ok) {
          const data = await response.json();
          
          dispatch({
            type: AuthActionTypes.AUTH_SUCCESS,
            payload: {
              user,
              accessToken: storedToken,
              refreshToken: storedRefreshToken,
              tokenExpiry: data.tokenExpiry,
              sessionId: data.sessionId
            }
          });

          // Load user profile and preferences
          await loadUserProfile();
          await loadUserPreferences();
          
          // Sync with AppContext
          appActions.setUser(user);
          appActions.setAuthenticated(true);
        } else {
          // Token invalid, try refresh
          await refreshAccessToken();
        }
      }
    } catch (error) {
      console.error('Auth initialization error:', error);
    } finally {
      dispatch({ type: AuthActionTypes.AUTH_INITIALIZED });
    }
  }, [API_BASE_URL, appActions]);

  // Login function
  const login = useCallback(async (credentials) => {
    try {
      dispatch({ type: AuthActionTypes.AUTH_START });

      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...credentials,
          deviceInfo: getDeviceInfo()
        })
      });

      const data = await response.json();

      if (response.ok) {
        // Store tokens
        localStorage.setItem('football-analytics-token', data.accessToken);
        localStorage.setItem('football-analytics-refresh-token', data.refreshToken);
        localStorage.setItem('football-analytics-user', JSON.stringify(data.user));

        dispatch({
          type: AuthActionTypes.AUTH_SUCCESS,
          payload: {
            user: data.user,
            accessToken: data.accessToken,
            refreshToken: data.refreshToken,
            tokenExpiry: data.tokenExpiry,
            sessionId: data.sessionId
          }
        });

        // Load additional user data
        await loadUserProfile();
        await loadUserPreferences();
        
        // Sync with AppContext
        appActions.setUser(data.user);
        appActions.setAuthenticated(true);
        appActions.clearAllErrors();

        // Track analytics
        dispatch({
          type: AuthActionTypes.UPDATE_ANALYTICS,
          payload: {
            loginCount: authState.analytics.loginCount + 1,
            lastLogin: new Date()
          }
        });

        return { success: true };
      } else {
        throw new Error(data.message || 'Login failed');
      }
    } catch (error) {
      dispatch({
        type: AuthActionTypes.AUTH_FAILURE,
        payload: error.message
      });
      
      appActions.setError('auth', error.message);
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL, appActions, authState.analytics.loginCount]);

  // Register function
  const register = useCallback(async (userData) => {
    try {
      dispatch({ type: AuthActionTypes.AUTH_START });

      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...userData,
          deviceInfo: getDeviceInfo()
        })
      });

      const data = await response.json();

      if (response.ok) {
        // Auto-login after registration
        return await login({
          email: userData.email,
          password: userData.password
        });
      } else {
        throw new Error(data.message || 'Registration failed');
      }
    } catch (error) {
      dispatch({
        type: AuthActionTypes.AUTH_FAILURE,
        payload: error.message
      });
      
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL, login]);

  // Logout function
  const logout = useCallback(async (allDevices = false) => {
    try {
      if (authState.accessToken) {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authState.accessToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            allDevices,
            sessionId: authState.sessionId
          })
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage
      localStorage.removeItem('football-analytics-token');
      localStorage.removeItem('football-analytics-refresh-token');
      localStorage.removeItem('football-analytics-user');

      dispatch({ type: AuthActionTypes.AUTH_LOGOUT });
      
      // Sync with AppContext
      appActions.logout();
    }
  }, [API_BASE_URL, authState.accessToken, authState.sessionId, appActions]);

  // Refresh access token
  const refreshAccessToken = useCallback(async () => {
    try {
      const refreshToken = localStorage.getItem('football-analytics-refresh-token');
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ refreshToken })
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('football-analytics-token', data.accessToken);
        
        dispatch({
          type: AuthActionTypes.REFRESH_TOKEN_SUCCESS,
          payload: {
            accessToken: data.accessToken,
            tokenExpiry: data.tokenExpiry
          }
        });

        return data.accessToken;
      } else {
        throw new Error(data.message || 'Token refresh failed');
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      dispatch({ type: AuthActionTypes.REFRESH_TOKEN_FAILURE });
      await logout();
      return null;
    }
  }, [API_BASE_URL, logout]);

  // Load user profile
  const loadUserProfile = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/profile`, {
        headers: {
          'Authorization': `Bearer ${authState.accessToken}`
        }
      });

      if (response.ok) {
        const profile = await response.json();
        dispatch({
          type: AuthActionTypes.UPDATE_PROFILE,
          payload: profile
        });
      }
    } catch (error) {
      console.error('Load profile error:', error);
    }
  }, [API_BASE_URL, authState.accessToken]);

  // Load user preferences
  const loadUserPreferences = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/preferences`, {
        headers: {
          'Authorization': `Bearer ${authState.accessToken}`
        }
      });

      if (response.ok) {
        const preferences = await response.json();
        dispatch({
          type: AuthActionTypes.SET_PREFERENCES,
          payload: preferences
        });
      }
    } catch (error) {
      console.error('Load preferences error:', error);
    }
  }, [API_BASE_URL, authState.accessToken]);

  // Update user profile
  const updateProfile = useCallback(async (profileData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/profile`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${authState.accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(profileData)
      });

      if (response.ok) {
        const updatedProfile = await response.json();
        dispatch({
          type: AuthActionTypes.UPDATE_PROFILE,
          payload: updatedProfile
        });
        return { success: true };
      } else {
        const data = await response.json();
        throw new Error(data.message || 'Profile update failed');
      }
    } catch (error) {
      console.error('Update profile error:', error);
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL, authState.accessToken]);

  // Update user preferences
  const updatePreferences = useCallback(async (key, value) => {
    try {
      const updatedPreferences = {
        ...authState.preferences,
        [key]: value
      };

      const response = await fetch(`${API_BASE_URL}/user/preferences`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${authState.accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedPreferences)
      });

      if (response.ok) {
        dispatch({
          type: AuthActionTypes.UPDATE_PREFERENCE,
          payload: { key, value }
        });
        return { success: true };
      } else {
        throw new Error('Preferences update failed');
      }
    } catch (error) {
      console.error('Update preferences error:', error);
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL, authState.accessToken, authState.preferences]);

  // Change password
  const changePassword = useCallback(async (currentPassword, newPassword) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/change-password`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authState.accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          currentPassword,
          newPassword
        })
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        throw new Error(data.message || 'Password change failed');
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL, authState.accessToken]);

  // Forgot password
  const forgotPassword = useCallback(async (email) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        throw new Error(data.message || 'Password reset failed');
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL]);

  // Reset password
  const resetPassword = useCallback(async (token, newPassword) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          token,
          newPassword
        })
      });

      const data = await response.json();

      if (response.ok) {
        return { success: true };
      } else {
        throw new Error(data.message || 'Password reset failed');
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }, [API_BASE_URL]);

  // Update activity (for session management)
  const updateActivity = useCallback(() => {
    if (authState.isAuthenticated) {
      dispatch({ type: AuthActionTypes.UPDATE_ACTIVITY });
    }
  }, [authState.isAuthenticated]);

  // Handle session timeout
  const handleSessionTimeout = useCallback(() => {
    dispatch({ type: AuthActionTypes.SESSION_TIMEOUT });
    appActions.setError('auth', 'Session timed out. Please login again.');
  }, [appActions]);

  // Track feature usage
  const trackFeatureUsage = useCallback((feature) => {
    if (authState.isAuthenticated) {
      dispatch({
        type: AuthActionTypes.TRACK_FEATURE_USAGE,
        payload: feature
      });
    }
  }, [authState.isAuthenticated]);

  // Get device info
  const getDeviceInfo = useCallback(() => {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      screenResolution: `${screen.width}x${screen.height}`,
      timestamp: new Date().toISOString()
    };
  }, []);

  // Get authenticated API headers
  const getAuthHeaders = useCallback(() => {
    const headers = {
      'Content-Type': 'application/json'
    };

    if (authState.accessToken) {
      headers['Authorization'] = `Bearer ${authState.accessToken}`;
    }

    return headers;
  }, [authState.accessToken]);

  // Check if user has permission
  const hasPermission = useCallback((permission) => {
    return authState.permissions.includes(permission) || authState.roles.includes('admin');
  }, [authState.permissions, authState.roles]);

  // Check if user has subscription feature
  const hasFeature = useCallback((feature) => {
    return authState.subscription.features.includes(feature) || authState.subscription.type === 'pro';
  }, [authState.subscription.features, authState.subscription.type]);

  // Auth context value
  const value = {
    // State
    ...authState,
    
    // Actions
    login,
    register,
    logout,
    refreshAccessToken,
    loadUserProfile,
    loadUserPreferences,
    updateProfile,
    updatePreferences,
    changePassword,
    forgotPassword,
    resetPassword,
    updateActivity,
    trackFeatureUsage,
    
    // Utilities
    getAuthHeaders,
    hasPermission,
    hasFeature,
    getDeviceInfo,
    
    // Advanced actions (can be implemented later)
    enableTwoFactor: () => console.log('2FA not implemented yet'),
    verifyTwoFactor: () => console.log('2FA verification not implemented yet'),
    addDevice: () => console.log('Device management not implemented yet'),
    removeDevice: () => console.log('Device removal not implemented yet')
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// HOC for protected routes
export const withAuth = (Component, permissions = []) => {
  return (props) => {
    const { isAuthenticated, hasPermission, isLoading } = useAuth();
    
    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
        </div>
      );
    }
    
    if (!isAuthenticated) {
      // Redirect to login or show login form
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Authentication Required
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Please login to access this page.
            </p>
          </div>
        </div>
      );
    }
    
    if (permissions.length > 0 && !permissions.some(permission => hasPermission(permission))) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Access Denied
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              You don't have permission to access this page.
            </p>
          </div>
        </div>
      );
    }
    
    return <Component {...props} />;
  };
};

export default AuthContext;