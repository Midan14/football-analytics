// Authentication Service for Football Analytics Platform
// Handles all authentication logic, token management, and security features

import { apiUtils, authAPI, userAPI } from './api';

// Auth configuration
const AUTH_CONFIG = {
  tokenKey: 'football-analytics-token',
  refreshTokenKey: 'football-analytics-refresh-token',
  userKey: 'football-analytics-user',
  sessionKey: 'football-analytics-session',
  rememberMeKey: 'football-analytics-remember-me',
  deviceKey: 'football-analytics-device-id',
  
  // Token expiry buffer (refresh 5 minutes before expiry)
  tokenRefreshBuffer: 5 * 60 * 1000,
  
  // Session timeout (30 minutes of inactivity)
  sessionTimeout: 30 * 60 * 1000,
  
  // Password requirements
  passwordRequirements: {
    minLength: 8,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: true,
    maxAge: 90 * 24 * 60 * 60 * 1000, // 90 days
  },
  
  // Login attempt limits
  maxLoginAttempts: 5,
  lockoutDuration: 15 * 60 * 1000, // 15 minutes
  
  // Two-factor authentication
  twoFactorEnabled: false,
  twoFactorMethods: ['sms', 'email', 'authenticator'],
  
  // Device management
  maxDevices: 5,
  deviceTrustDuration: 30 * 24 * 60 * 60 * 1000, // 30 days
};

// User roles and permissions
export const USER_ROLES = {
  ADMIN: 'admin',
  MODERATOR: 'moderator',
  PREMIUM: 'premium',
  USER: 'user',
  GUEST: 'guest'
};

export const PERMISSIONS = {
  // Admin permissions
  ADMIN_ACCESS: 'admin_access',
  USER_MANAGEMENT: 'user_management',
  SYSTEM_CONFIG: 'system_config',
  DATA_EXPORT: 'data_export',
  
  // Content permissions
  CREATE_CONTENT: 'create_content',
  EDIT_CONTENT: 'edit_content',
  DELETE_CONTENT: 'delete_content',
  MODERATE_CONTENT: 'moderate_content',
  
  // Analytics permissions
  VIEW_ANALYTICS: 'view_analytics',
  ADVANCED_ANALYTICS: 'advanced_analytics',
  EXPORT_DATA: 'export_data',
  
  // Prediction permissions
  VIEW_PREDICTIONS: 'view_predictions',
  ADVANCED_PREDICTIONS: 'advanced_predictions',
  PREDICTION_HISTORY: 'prediction_history',
  
  // API permissions
  API_ACCESS: 'api_access',
  RATE_LIMIT_EXTENDED: 'rate_limit_extended',
};

// Role-permission mapping
const ROLE_PERMISSIONS = {
  [USER_ROLES.ADMIN]: Object.values(PERMISSIONS),
  [USER_ROLES.MODERATOR]: [
    PERMISSIONS.VIEW_ANALYTICS,
    PERMISSIONS.MODERATE_CONTENT,
    PERMISSIONS.CREATE_CONTENT,
    PERMISSIONS.EDIT_CONTENT,
    PERMISSIONS.VIEW_PREDICTIONS,
    PERMISSIONS.API_ACCESS
  ],
  [USER_ROLES.PREMIUM]: [
    PERMISSIONS.VIEW_ANALYTICS,
    PERMISSIONS.ADVANCED_ANALYTICS,
    PERMISSIONS.VIEW_PREDICTIONS,
    PERMISSIONS.ADVANCED_PREDICTIONS,
    PERMISSIONS.PREDICTION_HISTORY,
    PERMISSIONS.EXPORT_DATA,
    PERMISSIONS.API_ACCESS,
    PERMISSIONS.RATE_LIMIT_EXTENDED
  ],
  [USER_ROLES.USER]: [
    PERMISSIONS.VIEW_ANALYTICS,
    PERMISSIONS.VIEW_PREDICTIONS,
    PERMISSIONS.API_ACCESS
  ],
  [USER_ROLES.GUEST]: []
};

// Auth event types
export const AUTH_EVENTS = {
  LOGIN_SUCCESS: 'login_success',
  LOGIN_FAILURE: 'login_failure',
  LOGOUT: 'logout',
  TOKEN_REFRESHED: 'token_refreshed',
  TOKEN_EXPIRED: 'token_expired',
  SESSION_TIMEOUT: 'session_timeout',
  PASSWORD_CHANGED: 'password_changed',
  ACCOUNT_LOCKED: 'account_locked',
  TWO_FACTOR_ENABLED: 'two_factor_enabled',
  DEVICE_TRUSTED: 'device_trusted'
};

// Auth errors
export const AUTH_ERRORS = {
  INVALID_CREDENTIALS: 'invalid_credentials',
  ACCOUNT_LOCKED: 'account_locked',
  ACCOUNT_DISABLED: 'account_disabled',
  EMAIL_NOT_VERIFIED: 'email_not_verified',
  TWO_FACTOR_REQUIRED: 'two_factor_required',
  PASSWORD_EXPIRED: 'password_expired',
  SESSION_EXPIRED: 'session_expired',
  DEVICE_NOT_TRUSTED: 'device_not_trusted',
  RATE_LIMITED: 'rate_limited'
};

class AuthService {
  constructor() {
    this.user = null;
    this.tokens = {
      access: null,
      refresh: null,
      expiresAt: null
    };
    this.session = {
      id: null,
      lastActivity: null,
      deviceId: null
    };
    this.eventCallbacks = new Map();
    this.refreshTimer = null;
    this.sessionTimer = null;
    this.isRefreshing = false;
    
    // Initialize from storage
    this.initializeFromStorage();
    
    // Set up automatic token refresh
    this.setupTokenRefresh();
    
    // Set up session monitoring
    this.setupSessionMonitoring();
    
    // Set up API auth event listener
    apiUtils.onAuthEvent((event, data) => {
      this.handleAPIAuthEvent(event, data);
    });
  }

  // Initialize auth state from localStorage
  initializeFromStorage() {
    try {
      const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
      const refreshToken = localStorage.getItem(AUTH_CONFIG.refreshTokenKey);
      const userData = localStorage.getItem(AUTH_CONFIG.userKey);
      const sessionData = localStorage.getItem(AUTH_CONFIG.sessionKey);

      if (token && userData) {
        this.tokens.access = token;
        this.tokens.refresh = refreshToken;
        this.user = JSON.parse(userData);
        
        if (sessionData) {
          this.session = JSON.parse(sessionData);
        }

        // Set tokens in API client
        apiUtils.setAuthTokens(token, refreshToken);
        
        // Verify token validity
        this.verifyToken();
      }
    } catch (error) {
      console.error('Failed to initialize auth from storage:', error);
      this.clearAuthData();
    }
  }

  // Setup automatic token refresh
  setupTokenRefresh() {
    const scheduleRefresh = () => {
      if (!this.tokens.access || !this.tokens.expiresAt) return;
      
      const now = Date.now();
      const expiresAt = new Date(this.tokens.expiresAt).getTime();
      const refreshTime = expiresAt - AUTH_CONFIG.tokenRefreshBuffer;
      const timeUntilRefresh = refreshTime - now;

      if (timeUntilRefresh > 0) {
        this.refreshTimer = setTimeout(() => {
          this.refreshToken();
        }, timeUntilRefresh);
      } else {
        // Token is already close to expiry, refresh immediately
        this.refreshToken();
      }
    };

    scheduleRefresh();
  }

  // Setup session monitoring
  setupSessionMonitoring() {
    const checkSession = () => {
      if (!this.isAuthenticated()) return;
      
      const lastActivity = this.session.lastActivity ? 
        new Date(this.session.lastActivity).getTime() : 
        Date.now();
      
      const timeSinceActivity = Date.now() - lastActivity;
      
      if (timeSinceActivity > AUTH_CONFIG.sessionTimeout) {
        this.handleSessionTimeout();
      }
    };

    // Check session every minute
    this.sessionTimer = setInterval(checkSession, 60 * 1000);
    
    // Update last activity on user interaction
    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
    const updateActivity = () => this.updateLastActivity();
    
    activityEvents.forEach(event => {
      document.addEventListener(event, updateActivity, { passive: true });
    });
  }

  // Handle API auth events
  handleAPIAuthEvent(event, data) {
    switch (event) {
      case 'token_refreshed':
        this.tokens.access = data.accessToken;
        this.tokens.expiresAt = data.expiresAt;
        this.saveToStorage();
        this.emit(AUTH_EVENTS.TOKEN_REFRESHED, data);
        break;
        
      case 'logout':
        this.logout();
        break;
        
      default:
        break;
    }
  }

  // Event emission system
  on(event, callback) {
    if (!this.eventCallbacks.has(event)) {
      this.eventCallbacks.set(event, new Set());
    }
    this.eventCallbacks.get(event).add(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.eventCallbacks.get(event);
      if (callbacks) {
        callbacks.delete(callback);
      }
    };
  }

  off(event, callback) {
    const callbacks = this.eventCallbacks.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  emit(event, data = null) {
    const callbacks = this.eventCallbacks.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Auth event callback error:', error);
        }
      });
    }
  }

  // Login method
  async login(credentials, options = {}) {
    try {
      const {
        rememberMe = false,
        deviceName = this.getDeviceInfo().name,
        trustDevice = false
      } = options;

      // Check for rate limiting
      if (this.isRateLimited()) {
        throw new Error(AUTH_ERRORS.RATE_LIMITED);
      }

      // Prepare login data
      const loginData = {
        ...credentials,
        deviceInfo: this.getDeviceInfo(),
        deviceName,
        trustDevice,
        rememberMe
      };

      const response = await authAPI.login(loginData);

      if (response.success) {
        const { user, accessToken, refreshToken, expiresAt, sessionId } = response.data;
        
        // Store auth data
        this.user = user;
        this.tokens = {
          access: accessToken,
          refresh: refreshToken,
          expiresAt
        };
        this.session = {
          id: sessionId,
          lastActivity: new Date().toISOString(),
          deviceId: this.getDeviceId()
        };

        // Set tokens in API client
        apiUtils.setAuthTokens(accessToken, refreshToken);
        
        // Save to storage
        this.saveToStorage();
        
        // Set remember me preference
        if (rememberMe) {
          localStorage.setItem(AUTH_CONFIG.rememberMeKey, 'true');
        }

        // Setup automatic refresh
        this.setupTokenRefresh();
        
        // Reset login attempts
        this.clearLoginAttempts();
        
        // Track login analytics
        this.trackLoginSuccess();
        
        // Emit login success event
        this.emit(AUTH_EVENTS.LOGIN_SUCCESS, { user, sessionId });
        
        return { success: true, user };
      } else {
        // Handle login failure
        this.trackLoginFailure(response.error);
        this.emit(AUTH_EVENTS.LOGIN_FAILURE, response.error);
        
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Login error:', error);
      this.trackLoginFailure(error);
      this.emit(AUTH_EVENTS.LOGIN_FAILURE, error);
      
      return { 
        success: false, 
        error: { 
          message: error.message || 'Login failed',
          code: error.code || AUTH_ERRORS.INVALID_CREDENTIALS
        }
      };
    }
  }

  // Register method
  async register(userData, options = {}) {
    try {
      // Validate password requirements
      const passwordValidation = this.validatePassword(userData.password);
      if (!passwordValidation.isValid) {
        return {
          success: false,
          error: {
            message: 'Password does not meet requirements',
            code: AUTH_ERRORS.INVALID_CREDENTIALS,
            details: passwordValidation.errors
          }
        };
      }

      const registrationData = {
        ...userData,
        deviceInfo: this.getDeviceInfo(),
        acceptedTerms: options.acceptedTerms || false,
        marketingConsent: options.marketingConsent || false
      };

      const response = await authAPI.register(registrationData);

      if (response.success) {
        // Auto-login after successful registration
        return this.login({
          email: userData.email,
          password: userData.password
        }, options);
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Registration error:', error);
      return { 
        success: false, 
        error: { 
          message: error.message || 'Registration failed',
          code: AUTH_ERRORS.INVALID_CREDENTIALS
        }
      };
    }
  }

  // Logout method
  async logout(options = {}) {
    try {
      const { allDevices = false, reason = 'user_logout' } = options;
      
      // Call logout API if authenticated
      if (this.isAuthenticated()) {
        await authAPI.logout().catch(error => {
          console.warn('Logout API call failed:', error);
        });
      }

      // Clear local auth data
      this.clearAuthData();
      
      // Clear timers
      if (this.refreshTimer) {
        clearTimeout(this.refreshTimer);
        this.refreshTimer = null;
      }
      
      if (this.sessionTimer) {
        clearInterval(this.sessionTimer);
        this.sessionTimer = null;
      }

      // Track logout
      this.trackLogout(reason);
      
      // Emit logout event
      this.emit(AUTH_EVENTS.LOGOUT, { reason, allDevices });
      
      return { success: true };
    } catch (error) {
      console.error('Logout error:', error);
      
      // Force clear even if API call failed
      this.clearAuthData();
      
      return { 
        success: false, 
        error: { message: error.message || 'Logout failed' }
      };
    }
  }

  // Refresh token method
  async refreshToken() {
    if (this.isRefreshing) return null;
    
    try {
      this.isRefreshing = true;
      
      const response = await authAPI.refreshToken();
      
      if (response.success) {
        const { accessToken, expiresAt } = response.data;
        
        this.tokens.access = accessToken;
        this.tokens.expiresAt = expiresAt;
        
        // Update in API client
        apiUtils.setAuthTokens(accessToken, this.tokens.refresh);
        
        // Save to storage
        this.saveToStorage();
        
        // Schedule next refresh
        this.setupTokenRefresh();
        
        // Emit token refreshed event
        this.emit(AUTH_EVENTS.TOKEN_REFRESHED, response.data);
        
        return accessToken;
      } else {
        // Refresh failed, logout user
        this.logout({ reason: 'token_refresh_failed' });
        return null;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      this.logout({ reason: 'token_refresh_error' });
      return null;
    } finally {
      this.isRefreshing = false;
    }
  }

  // Verify token validity
  async verifyToken() {
    try {
      const response = await authAPI.verifyToken();
      
      if (response.success) {
        // Update token expiry if provided
        if (response.data.expiresAt) {
          this.tokens.expiresAt = response.data.expiresAt;
          this.saveToStorage();
        }
        
        return true;
      } else {
        // Token invalid, clear auth data
        this.clearAuthData();
        return false;
      }
    } catch (error) {
      console.error('Token verification error:', error);
      this.clearAuthData();
      return false;
    }
  }

  // Password change method
  async changePassword(currentPassword, newPassword) {
    try {
      // Validate new password
      const passwordValidation = this.validatePassword(newPassword);
      if (!passwordValidation.isValid) {
        return {
          success: false,
          error: {
            message: 'New password does not meet requirements',
            details: passwordValidation.errors
          }
        };
      }

      const response = await authAPI.changePassword(currentPassword, newPassword);
      
      if (response.success) {
        // Emit password changed event
        this.emit(AUTH_EVENTS.PASSWORD_CHANGED);
        
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Password change error:', error);
      return { 
        success: false, 
        error: { message: error.message || 'Password change failed' }
      };
    }
  }

  // Password reset methods
  async forgotPassword(email) {
    try {
      const response = await authAPI.forgotPassword(email);
      return response;
    } catch (error) {
      console.error('Forgot password error:', error);
      return { 
        success: false, 
        error: { message: error.message || 'Password reset request failed' }
      };
    }
  }

  async resetPassword(token, newPassword) {
    try {
      // Validate new password
      const passwordValidation = this.validatePassword(newPassword);
      if (!passwordValidation.isValid) {
        return {
          success: false,
          error: {
            message: 'Password does not meet requirements',
            details: passwordValidation.errors
          }
        };
      }

      const response = await authAPI.resetPassword(token, newPassword);
      return response;
    } catch (error) {
      console.error('Reset password error:', error);
      return { 
        success: false, 
        error: { message: error.message || 'Password reset failed' }
      };
    }
  }

  // User profile methods
  async updateProfile(profileData) {
    try {
      const response = await userAPI.updateProfile(profileData);
      
      if (response.success) {
        // Update local user data
        this.user = { ...this.user, ...response.data };
        this.saveToStorage();
        
        return { success: true, user: this.user };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Profile update error:', error);
      return { 
        success: false, 
        error: { message: error.message || 'Profile update failed' }
      };
    }
  }

  // Session management
  updateLastActivity() {
    if (this.isAuthenticated()) {
      this.session.lastActivity = new Date().toISOString();
      this.saveSessionToStorage();
    }
  }

  handleSessionTimeout() {
    this.logout({ reason: 'session_timeout' });
    this.emit(AUTH_EVENTS.SESSION_TIMEOUT);
  }

  extendSession() {
    this.updateLastActivity();
  }

  // Permission and role checking
  hasRole(role) {
    return this.user?.roles?.includes(role) || false;
  }

  hasPermission(permission) {
    if (!this.user?.roles) return false;
    
    // Admin has all permissions
    if (this.hasRole(USER_ROLES.ADMIN)) return true;
    
    // Check role permissions
    for (const role of this.user.roles) {
      const rolePermissions = ROLE_PERMISSIONS[role] || [];
      if (rolePermissions.includes(permission)) {
        return true;
      }
    }
    
    // Check direct permissions
    return this.user.permissions?.includes(permission) || false;
  }

  hasAnyRole(roles) {
    return roles.some(role => this.hasRole(role));
  }

  hasAnyPermission(permissions) {
    return permissions.some(permission => this.hasPermission(permission));
  }

  // Subscription checking
  hasSubscription(type) {
    return this.user?.subscription?.type === type;
  }

  isSubscriptionActive() {
    if (!this.user?.subscription) return false;
    
    const expiresAt = this.user.subscription.expiresAt;
    if (!expiresAt) return true; // Lifetime subscription
    
    return new Date(expiresAt) > new Date();
  }

  getSubscriptionFeatures() {
    return this.user?.subscription?.features || [];
  }

  hasFeature(feature) {
    const features = this.getSubscriptionFeatures();
    return features.includes(feature);
  }

  // Device management
  getDeviceInfo() {
    return {
      id: this.getDeviceId(),
      name: this.getDeviceName(),
      type: this.getDeviceType(),
      os: this.getOS(),
      browser: this.getBrowser(),
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString()
    };
  }

  getDeviceId() {
    let deviceId = localStorage.getItem(AUTH_CONFIG.deviceKey);
    if (!deviceId) {
      deviceId = this.generateDeviceId();
      localStorage.setItem(AUTH_CONFIG.deviceKey, deviceId);
    }
    return deviceId;
  }

  generateDeviceId() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillText('Device fingerprint', 2, 2);
    
    const fingerprint = canvas.toDataURL();
    const hash = this.simpleHash(fingerprint + navigator.userAgent);
    
    return `device_${hash}_${Date.now()}`;
  }

  getDeviceName() {
    const os = this.getOS();
    const browser = this.getBrowser();
    return `${os} - ${browser}`;
  }

  getDeviceType() {
    const userAgent = navigator.userAgent.toLowerCase();
    if (/mobile|android|iphone|ipad|phone/i.test(userAgent)) {
      return 'mobile';
    } else if (/tablet|ipad/i.test(userAgent)) {
      return 'tablet';
    } else {
      return 'desktop';
    }
  }

  getOS() {
    const userAgent = navigator.userAgent;
    if (userAgent.includes('Windows')) return 'Windows';
    if (userAgent.includes('Mac')) return 'macOS';
    if (userAgent.includes('Linux')) return 'Linux';
    if (userAgent.includes('Android')) return 'Android';
    if (userAgent.includes('iOS')) return 'iOS';
    return 'Unknown';
  }

  getBrowser() {
    const userAgent = navigator.userAgent;
    if (userAgent.includes('Chrome')) return 'Chrome';
    if (userAgent.includes('Firefox')) return 'Firefox';
    if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) return 'Safari';
    if (userAgent.includes('Edge')) return 'Edge';
    return 'Unknown';
  }

  // Password validation
  validatePassword(password) {
    const errors = [];
    const config = AUTH_CONFIG.passwordRequirements;
    
    if (password.length < config.minLength) {
      errors.push(`Password must be at least ${config.minLength} characters long`);
    }
    
    if (config.requireUppercase && !/[A-Z]/.test(password)) {
      errors.push('Password must contain at least one uppercase letter');
    }
    
    if (config.requireLowercase && !/[a-z]/.test(password)) {
      errors.push('Password must contain at least one lowercase letter');
    }
    
    if (config.requireNumbers && !/\d/.test(password)) {
      errors.push('Password must contain at least one number');
    }
    
    if (config.requireSpecialChars && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      errors.push('Password must contain at least one special character');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  }

  // Rate limiting
  isRateLimited() {
    const attempts = this.getLoginAttempts();
    const lockoutTime = localStorage.getItem('auth_lockout_until');
    
    if (lockoutTime && Date.now() < parseInt(lockoutTime)) {
      return true;
    }
    
    return attempts >= AUTH_CONFIG.maxLoginAttempts;
  }

  getLoginAttempts() {
    const attempts = localStorage.getItem('auth_login_attempts');
    return attempts ? parseInt(attempts) : 0;
  }

  incrementLoginAttempts() {
    const attempts = this.getLoginAttempts() + 1;
    localStorage.setItem('auth_login_attempts', attempts.toString());
    
    if (attempts >= AUTH_CONFIG.maxLoginAttempts) {
      const lockoutUntil = Date.now() + AUTH_CONFIG.lockoutDuration;
      localStorage.setItem('auth_lockout_until', lockoutUntil.toString());
      this.emit(AUTH_EVENTS.ACCOUNT_LOCKED);
    }
  }

  clearLoginAttempts() {
    localStorage.removeItem('auth_login_attempts');
    localStorage.removeItem('auth_lockout_until');
  }

  // Analytics tracking
  trackLoginSuccess() {
    this.trackAuthEvent('login_success', {
      timestamp: new Date().toISOString(),
      deviceInfo: this.getDeviceInfo()
    });
  }

  trackLoginFailure(error) {
    this.incrementLoginAttempts();
    this.trackAuthEvent('login_failure', {
      timestamp: new Date().toISOString(),
      error: error?.message || 'Unknown error',
      deviceInfo: this.getDeviceInfo()
    });
  }

  trackLogout(reason) {
    this.trackAuthEvent('logout', {
      timestamp: new Date().toISOString(),
      reason,
      sessionDuration: this.getSessionDuration()
    });
  }

  trackAuthEvent(event, data) {
    try {
      // Send to analytics service
      if (window.gtag) {
        window.gtag('event', event, data);
      }
      
      // Store locally for debugging
      if (process.env.NODE_ENV === 'development') {
        console.log('Auth Event:', event, data);
      }
    } catch (error) {
      console.error('Failed to track auth event:', error);
    }
  }

  getSessionDuration() {
    if (!this.session.lastActivity) return 0;
    return Date.now() - new Date(this.session.lastActivity).getTime();
  }

  // Utility methods
  isAuthenticated() {
    return !!(this.user && this.tokens.access);
  }

  getUser() {
    return this.user;
  }

  getToken() {
    return this.tokens.access;
  }

  getTokenExpiresAt() {
    return this.tokens.expiresAt;
  }

  getSession() {
    return this.session;
  }

  isTokenExpired() {
    if (!this.tokens.expiresAt) return false;
    return Date.now() >= new Date(this.tokens.expiresAt).getTime();
  }

  willTokenExpireSoon(buffer = AUTH_CONFIG.tokenRefreshBuffer) {
    if (!this.tokens.expiresAt) return false;
    return Date.now() >= (new Date(this.tokens.expiresAt).getTime() - buffer);
  }

  // Storage methods
  saveToStorage() {
    try {
      if (this.tokens.access) {
        localStorage.setItem(AUTH_CONFIG.tokenKey, this.tokens.access);
      }
      
      if (this.tokens.refresh) {
        localStorage.setItem(AUTH_CONFIG.refreshTokenKey, this.tokens.refresh);
      }
      
      if (this.user) {
        localStorage.setItem(AUTH_CONFIG.userKey, JSON.stringify(this.user));
      }
      
      this.saveSessionToStorage();
    } catch (error) {
      console.error('Failed to save auth data to storage:', error);
    }
  }

  saveSessionToStorage() {
    try {
      if (this.session) {
        localStorage.setItem(AUTH_CONFIG.sessionKey, JSON.stringify(this.session));
      }
    } catch (error) {
      console.error('Failed to save session data to storage:', error);
    }
  }

  clearAuthData() {
    this.user = null;
    this.tokens = { access: null, refresh: null, expiresAt: null };
    this.session = { id: null, lastActivity: null, deviceId: null };
    
    // Clear from localStorage
    localStorage.removeItem(AUTH_CONFIG.tokenKey);
    localStorage.removeItem(AUTH_CONFIG.refreshTokenKey);
    localStorage.removeItem(AUTH_CONFIG.userKey);
    localStorage.removeItem(AUTH_CONFIG.sessionKey);
    
    // Clear from API client
    apiUtils.clearAuthTokens();
  }

  // Simple hash function for device fingerprinting
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  // Cleanup method
  destroy() {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }
    
    if (this.sessionTimer) {
      clearInterval(this.sessionTimer);
    }
    
    this.eventCallbacks.clear();
  }
}

// Create singleton instance
const authService = new AuthService();

// Export the service instance and utilities
export default authService;

export {
    AUTH_CONFIG, AUTH_ERRORS, AUTH_EVENTS, AuthService, PERMISSIONS, USER_ROLES
};
