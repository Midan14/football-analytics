// API Service for Football Analytics Platform
// Centralized API management with interceptors, caching, and error handling

// Configuration
const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000,
  cacheTimeout: 5 * 60 * 1000, // 5 minutes
  version: 'v1'
};

// API endpoints mapping
const ENDPOINTS = {
  // Authentication
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    VERIFY: '/auth/verify',
    FORGOT_PASSWORD: '/auth/forgot-password',
    RESET_PASSWORD: '/auth/reset-password',
    CHANGE_PASSWORD: '/auth/change-password'
  },

  // User Management
  USER: {
    PROFILE: '/user/profile',
    PREFERENCES: '/user/preferences',
    FAVORITES: {
      BASE: '/user/favorites',
      MATCHES: '/user/favorites/matches',
      PLAYERS: '/user/favorites/players',
      TEAMS: '/user/favorites/teams',
      LEAGUES: '/user/favorites/leagues',
      PREDICTIONS: '/user/favorites/predictions'
    },
    NOTIFICATIONS: '/user/notifications',
    ANALYTICS: '/user/analytics'
  },

  // Matches
  MATCHES: {
    BASE: '/matches',
    LIVE: '/matches/live',
    UPCOMING: '/matches/upcoming',
    FINISHED: '/matches/finished',
    BY_DATE: '/matches/date',
    BY_LEAGUE: '/matches/league',
    BY_TEAM: '/matches/team',
    SEARCH: '/matches/search',
    DETAIL: '/matches' // + /:id
  },

  // Players
  PLAYERS: {
    BASE: '/players',
    SEARCH: '/players/search',
    BY_TEAM: '/players/team',
    BY_POSITION: '/players/position',
    BY_LEAGUE: '/players/league',
    STATISTICS: '/players/statistics',
    DETAIL: '/players' // + /:id
  },

  // Teams
  TEAMS: {
    BASE: '/teams',
    SEARCH: '/teams/search',
    BY_LEAGUE: '/teams/league',
    BY_COUNTRY: '/teams/country',
    STATISTICS: '/teams/statistics',
    STANDINGS: '/teams/standings',
    DETAIL: '/teams' // + /:id
  },

  // Leagues
  LEAGUES: {
    BASE: '/leagues',
    SEARCH: '/leagues/search',
    BY_COUNTRY: '/leagues/country',
    BY_CONFEDERATION: '/leagues/confederation',
    STANDINGS: '/leagues/standings',
    STATISTICS: '/leagues/statistics',
    DETAIL: '/leagues' // + /:id
  },

  // Predictions
  PREDICTIONS: {
    BASE: '/predictions',
    BY_MATCH: '/predictions/match',
    BY_DATE: '/predictions/date',
    SIMILAR_MATCHES: '/predictions/similar-matches',
    MODEL_PERFORMANCE: '/predictions/model-performance',
    ACCURACY: '/predictions/accuracy',
    TRENDS: '/predictions/trends'
  },

  // Injuries
  INJURIES: {
    BASE: '/injuries',
    ACTIVE: '/injuries/active',
    BY_PLAYER: '/injuries/player',
    BY_TEAM: '/injuries/team',
    BY_TYPE: '/injuries/type',
    STATISTICS: '/injuries/statistics',
    RECOVERY: '/injuries/recovery'
  },

  // Statistics
  STATISTICS: {
    GLOBAL: '/statistics/global',
    MATCHES: '/statistics/matches',
    PLAYERS: '/statistics/players',
    TEAMS: '/statistics/teams',
    LEAGUES: '/statistics/leagues',
    PERFORMANCE: '/statistics/performance'
  },

  // Reference Data
  REFERENCE: {
    COUNTRIES: '/reference/countries',
    CONFEDERATIONS: '/reference/confederations',
    POSITIONS: '/reference/positions',
    INJURY_TYPES: '/reference/injury-types',
    SEASONS: '/reference/seasons'
  }
};

// Error types
export const API_ERRORS = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR'
};

// HTTP status codes mapping
const STATUS_CODES = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504
};

// Simple in-memory cache
class APICache {
  constructor() {
    this.cache = new Map();
  }

  set(key, data, ttl = API_CONFIG.cacheTimeout) {
    const expiry = Date.now() + ttl;
    this.cache.set(key, { data, expiry });
  }

  get(key) {
    const item = this.cache.get(key);
    if (!item) return null;
    
    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data;
  }

  delete(key) {
    this.cache.delete(key);
  }

  clear() {
    this.cache.clear();
  }

  clearExpired() {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiry) {
        this.cache.delete(key);
      }
    }
  }
}

const apiCache = new APICache();

// Request queue to prevent duplicate requests
const requestQueue = new Map();

// API Client class
class APIClient {
  constructor() {
    this.authToken = null;
    this.refreshToken = null;
    this.authCallbacks = new Set();
    this.interceptors = {
      request: [],
      response: []
    };
    
    // Set up default request interceptor
    this.addRequestInterceptor(this.defaultRequestInterceptor.bind(this));
    
    // Set up default response interceptor
    this.addResponseInterceptor(this.defaultResponseInterceptor.bind(this));
  }

  // Authentication methods
  setAuthTokens(accessToken, refreshToken) {
    this.authToken = accessToken;
    this.refreshToken = refreshToken;
    localStorage.setItem('football-analytics-token', accessToken);
    if (refreshToken) {
      localStorage.setItem('football-analytics-refresh-token', refreshToken);
    }
  }

  clearAuthTokens() {
    this.authToken = null;
    this.refreshToken = null;
    localStorage.removeItem('football-analytics-token');
    localStorage.removeItem('football-analytics-refresh-token');
    apiCache.clear();
  }

  getAuthToken() {
    return this.authToken || localStorage.getItem('football-analytics-token');
  }

  getRefreshToken() {
    return this.refreshToken || localStorage.getItem('football-analytics-refresh-token');
  }

  // Interceptor methods
  addRequestInterceptor(interceptor) {
    this.interceptors.request.push(interceptor);
  }

  addResponseInterceptor(interceptor) {
    this.interceptors.response.push(interceptor);
  }

  // Default request interceptor
  defaultRequestInterceptor(config) {
    const token = this.getAuthToken();
    if (token) {
      config.headers = {
        ...config.headers,
        'Authorization': `Bearer ${token}`
      };
    }
    return config;
  }

  // Default response interceptor
  async defaultResponseInterceptor(response, config) {
    // Handle 401 Unauthorized - try to refresh token
    if (response.status === STATUS_CODES.UNAUTHORIZED && this.getRefreshToken()) {
      try {
        const refreshResponse = await this.refreshAuthToken();
        if (refreshResponse.success) {
          // Retry original request with new token
          return this.request(config.url, {
            ...config,
            _retry: true
          });
        }
      } catch (error) {
        // Refresh failed, clear tokens and redirect to login
        this.clearAuthTokens();
        this.notifyAuthCallbacks('logout');
        throw error;
      }
    }
    
    return response;
  }

  // Auth event management
  onAuthEvent(callback) {
    this.authCallbacks.add(callback);
    return () => this.authCallbacks.delete(callback);
  }

  notifyAuthCallbacks(event, data = null) {
    this.authCallbacks.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('Auth callback error:', error);
      }
    });
  }

  // Cache key generation
  generateCacheKey(url, options) {
    const params = new URLSearchParams();
    if (options.params) {
      Object.keys(options.params).sort().forEach(key => {
        params.append(key, options.params[key]);
      });
    }
    return `${options.method || 'GET'}:${url}?${params.toString()}`;
  }

  // Main request method
  async request(url, options = {}) {
    const config = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      cache: true,
      retry: true,
      timeout: API_CONFIG.timeout,
      ...options
    };

    // Build full URL
    const fullUrl = url.startsWith('http') ? url : `${API_CONFIG.baseURL}${url}`;
    
    // Add query parameters
    const urlWithParams = this.buildUrlWithParams(fullUrl, config.params);
    
    // Generate cache key
    const cacheKey = this.generateCacheKey(urlWithParams, config);
    
    // Check cache for GET requests
    if (config.method === 'GET' && config.cache && !config._retry) {
      const cachedData = apiCache.get(cacheKey);
      if (cachedData) {
        return { success: true, data: cachedData, cached: true };
      }
    }

    // Check for duplicate requests
    if (requestQueue.has(cacheKey) && !config._retry) {
      return requestQueue.get(cacheKey);
    }

    // Apply request interceptors
    for (const interceptor of this.interceptors.request) {
      config = await interceptor(config);
    }

    const requestPromise = this.executeRequest(urlWithParams, config, cacheKey);
    
    // Add to queue if not a retry
    if (!config._retry) {
      requestQueue.set(cacheKey, requestPromise);
    }

    try {
      const result = await requestPromise;
      return result;
    } finally {
      requestQueue.delete(cacheKey);
    }
  }

  // Execute the actual HTTP request
  async executeRequest(url, config, cacheKey) {
    let lastError = null;
    const maxRetries = config.retry ? API_CONFIG.retryAttempts : 0;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), config.timeout);

        // Prepare fetch options
        const fetchOptions = {
          method: config.method,
          headers: config.headers,
          signal: controller.signal
        };

        // Add body for non-GET requests
        if (config.method !== 'GET' && config.data) {
          fetchOptions.body = JSON.stringify(config.data);
        }

        // Execute request
        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);

        // Apply response interceptors
        let interceptedResponse = response;
        for (const interceptor of this.interceptors.response) {
          interceptedResponse = await interceptor(interceptedResponse, config);
        }

        // Parse response
        const result = await this.parseResponse(interceptedResponse);

        // Cache successful GET requests
        if (config.method === 'GET' && config.cache && result.success) {
          apiCache.set(cacheKey, result.data);
        }

        return result;

      } catch (error) {
        lastError = this.handleRequestError(error);

        // Don't retry certain errors
        if (this.shouldNotRetry(error) || attempt === maxRetries) {
          break;
        }

        // Wait before retry with exponential backoff
        const delay = API_CONFIG.retryDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  // Build URL with query parameters
  buildUrlWithParams(url, params) {
    if (!params) return url;
    
    const urlObj = new URL(url);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        urlObj.searchParams.append(key, value);
      }
    });
    
    return urlObj.toString();
  }

  // Parse response data
  async parseResponse(response) {
    try {
      const contentType = response.headers.get('content-type');
      let data;

      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      if (response.ok) {
        return {
          success: true,
          data,
          status: response.status,
          headers: response.headers
        };
      } else {
        return {
          success: false,
          error: {
            message: data.message || `HTTP ${response.status}: ${response.statusText}`,
            status: response.status,
            code: this.getErrorCode(response.status),
            data
          }
        };
      }
    } catch (error) {
      return {
        success: false,
        error: {
          message: 'Failed to parse response',
          code: API_ERRORS.UNKNOWN_ERROR,
          originalError: error
        }
      };
    }
  }

  // Handle request errors
  handleRequestError(error) {
    if (error.name === 'AbortError') {
      return {
        message: 'Request timeout',
        code: API_ERRORS.TIMEOUT_ERROR
      };
    }

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return {
        message: 'Network error',
        code: API_ERRORS.NETWORK_ERROR
      };
    }

    return {
      message: error.message || 'Unknown error',
      code: API_ERRORS.UNKNOWN_ERROR,
      originalError: error
    };
  }

  // Determine if request should not be retried
  shouldNotRetry(error) {
    return error.name === 'AbortError' || 
           error.message?.includes('401') || 
           error.message?.includes('403');
  }

  // Get error code from status
  getErrorCode(status) {
    switch (status) {
      case STATUS_CODES.UNAUTHORIZED:
        return API_ERRORS.UNAUTHORIZED;
      case STATUS_CODES.FORBIDDEN:
        return API_ERRORS.FORBIDDEN;
      case STATUS_CODES.NOT_FOUND:
        return API_ERRORS.NOT_FOUND;
      case STATUS_CODES.UNPROCESSABLE_ENTITY:
        return API_ERRORS.VALIDATION_ERROR;
      case STATUS_CODES.INTERNAL_SERVER_ERROR:
      case STATUS_CODES.BAD_GATEWAY:
      case STATUS_CODES.SERVICE_UNAVAILABLE:
      case STATUS_CODES.GATEWAY_TIMEOUT:
        return API_ERRORS.SERVER_ERROR;
      default:
        return API_ERRORS.UNKNOWN_ERROR;
    }
  }

  // HTTP method helpers
  async get(url, params = null, options = {}) {
    return this.request(url, { method: 'GET', params, ...options });
  }

  async post(url, data = null, options = {}) {
    return this.request(url, { method: 'POST', data, cache: false, ...options });
  }

  async put(url, data = null, options = {}) {
    return this.request(url, { method: 'PUT', data, cache: false, ...options });
  }

  async patch(url, data = null, options = {}) {
    return this.request(url, { method: 'PATCH', data, cache: false, ...options });
  }

  async delete(url, options = {}) {
    return this.request(url, { method: 'DELETE', cache: false, ...options });
  }

  // Refresh auth token
  async refreshAuthToken() {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await fetch(`${API_CONFIG.baseURL}${ENDPOINTS.AUTH.REFRESH}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ refreshToken })
      });

      if (response.ok) {
        const data = await response.json();
        this.setAuthTokens(data.accessToken, data.refreshToken);
        this.notifyAuthCallbacks('token_refreshed', data);
        return { success: true, data };
      } else {
        throw new Error('Token refresh failed');
      }
    } catch (error) {
      this.clearAuthTokens();
      throw error;
    }
  }
}

// Create singleton instance
const apiClient = new APIClient();

// Authentication API
export const authAPI = {
  login: (credentials) => 
    apiClient.post(ENDPOINTS.AUTH.LOGIN, credentials),
  
  register: (userData) => 
    apiClient.post(ENDPOINTS.AUTH.REGISTER, userData),
  
  logout: () => 
    apiClient.post(ENDPOINTS.AUTH.LOGOUT),
  
  verifyToken: () => 
    apiClient.get(ENDPOINTS.AUTH.VERIFY),
  
  forgotPassword: (email) => 
    apiClient.post(ENDPOINTS.AUTH.FORGOT_PASSWORD, { email }),
  
  resetPassword: (token, newPassword) => 
    apiClient.post(ENDPOINTS.AUTH.RESET_PASSWORD, { token, newPassword }),
  
  changePassword: (currentPassword, newPassword) => 
    apiClient.post(ENDPOINTS.AUTH.CHANGE_PASSWORD, { currentPassword, newPassword })
};

// User API
export const userAPI = {
  getProfile: () => 
    apiClient.get(ENDPOINTS.USER.PROFILE),
  
  updateProfile: (profileData) => 
    apiClient.put(ENDPOINTS.USER.PROFILE, profileData),
  
  getPreferences: () => 
    apiClient.get(ENDPOINTS.USER.PREFERENCES),
  
  updatePreferences: (preferences) => 
    apiClient.put(ENDPOINTS.USER.PREFERENCES, preferences),
  
  getNotifications: (params = {}) => 
    apiClient.get(ENDPOINTS.USER.NOTIFICATIONS, params),
  
  markNotificationRead: (notificationId) => 
    apiClient.patch(`${ENDPOINTS.USER.NOTIFICATIONS}/${notificationId}`, { read: true }),
  
  getAnalytics: () => 
    apiClient.get(ENDPOINTS.USER.ANALYTICS)
};

// Favorites API
export const favoritesAPI = {
  // Get favorites by type
  getMatches: () => 
    apiClient.get(ENDPOINTS.USER.FAVORITES.MATCHES),
  
  getPlayers: () => 
    apiClient.get(ENDPOINTS.USER.FAVORITES.PLAYERS),
  
  getTeams: () => 
    apiClient.get(ENDPOINTS.USER.FAVORITES.TEAMS),
  
  getLeagues: () => 
    apiClient.get(ENDPOINTS.USER.FAVORITES.LEAGUES),
  
  getPredictions: () => 
    apiClient.get(ENDPOINTS.USER.FAVORITES.PREDICTIONS),
  
  // Add favorites
  addMatch: (matchId, matchData = {}) => 
    apiClient.post(`${ENDPOINTS.USER.FAVORITES.MATCHES}/${matchId}`, matchData),
  
  addPlayer: (playerId, playerData = {}) => 
    apiClient.post(`${ENDPOINTS.USER.FAVORITES.PLAYERS}/${playerId}`, playerData),
  
  addTeam: (teamId, teamData = {}) => 
    apiClient.post(`${ENDPOINTS.USER.FAVORITES.TEAMS}/${teamId}`, teamData),
  
  addLeague: (leagueId, leagueData = {}) => 
    apiClient.post(`${ENDPOINTS.USER.FAVORITES.LEAGUES}/${leagueId}`, leagueData),
  
  // Remove favorites
  removeMatch: (matchId) => 
    apiClient.delete(`${ENDPOINTS.USER.FAVORITES.MATCHES}/${matchId}`),
  
  removePlayer: (playerId) => 
    apiClient.delete(`${ENDPOINTS.USER.FAVORITES.PLAYERS}/${playerId}`),
  
  removeTeam: (teamId) => 
    apiClient.delete(`${ENDPOINTS.USER.FAVORITES.TEAMS}/${teamId}`),
  
  removeLeague: (leagueId) => 
    apiClient.delete(`${ENDPOINTS.USER.FAVORITES.LEAGUES}/${leagueId}`)
};

// Matches API
export const matchesAPI = {
  getMatches: (params = {}) => 
    apiClient.get(ENDPOINTS.MATCHES.BASE, params),
  
  getLiveMatches: (params = {}) => 
    apiClient.get(ENDPOINTS.MATCHES.LIVE, params),
  
  getUpcomingMatches: (params = {}) => 
    apiClient.get(ENDPOINTS.MATCHES.UPCOMING, params),
  
  getFinishedMatches: (params = {}) => 
    apiClient.get(ENDPOINTS.MATCHES.FINISHED, params),
  
  getMatchesByDate: (date, params = {}) => 
    apiClient.get(`${ENDPOINTS.MATCHES.BY_DATE}/${date}`, params),
  
  getMatchesByLeague: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.MATCHES.BY_LEAGUE}/${leagueId}`, params),
  
  getMatchesByTeam: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.MATCHES.BY_TEAM}/${teamId}`, params),
  
  searchMatches: (query, params = {}) => 
    apiClient.get(ENDPOINTS.MATCHES.SEARCH, { q: query, ...params }),
  
  getMatchDetail: (matchId) => 
    apiClient.get(`${ENDPOINTS.MATCHES.DETAIL}/${matchId}`),
  
  getMatchStatistics: (matchId) => 
    apiClient.get(`${ENDPOINTS.MATCHES.DETAIL}/${matchId}/statistics`),
  
  getMatchLineups: (matchId) => 
    apiClient.get(`${ENDPOINTS.MATCHES.DETAIL}/${matchId}/lineups`),
  
  getMatchEvents: (matchId) => 
    apiClient.get(`${ENDPOINTS.MATCHES.DETAIL}/${matchId}/events`)
};

// Players API
export const playersAPI = {
  getPlayers: (params = {}) => 
    apiClient.get(ENDPOINTS.PLAYERS.BASE, params),
  
  searchPlayers: (query, params = {}) => 
    apiClient.get(ENDPOINTS.PLAYERS.SEARCH, { q: query, ...params }),
  
  getPlayersByTeam: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.BY_TEAM}/${teamId}`, params),
  
  getPlayersByPosition: (position, params = {}) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.BY_POSITION}/${position}`, params),
  
  getPlayersByLeague: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.BY_LEAGUE}/${leagueId}`, params),
  
  getPlayerDetail: (playerId) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.DETAIL}/${playerId}`),
  
  getPlayerStatistics: (playerId, params = {}) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.DETAIL}/${playerId}/statistics`, params),
  
  getPlayerMatches: (playerId, params = {}) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.DETAIL}/${playerId}/matches`, params),
  
  getPlayerInjuries: (playerId) => 
    apiClient.get(`${ENDPOINTS.PLAYERS.DETAIL}/${playerId}/injuries`)
};

// Teams API
export const teamsAPI = {
  getTeams: (params = {}) => 
    apiClient.get(ENDPOINTS.TEAMS.BASE, params),
  
  searchTeams: (query, params = {}) => 
    apiClient.get(ENDPOINTS.TEAMS.SEARCH, { q: query, ...params }),
  
  getTeamsByLeague: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.TEAMS.BY_LEAGUE}/${leagueId}`, params),
  
  getTeamsByCountry: (country, params = {}) => 
    apiClient.get(`${ENDPOINTS.TEAMS.BY_COUNTRY}/${country}`, params),
  
  getTeamDetail: (teamId) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}`),
  
  getTeamStatistics: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}/statistics`, params),
  
  getTeamPlayers: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}/players`, params),
  
  getTeamMatches: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}/matches`, params),
  
  getTeamInjuries: (teamId) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}/injuries`),
  
  getTeamStandings: (teamId) => 
    apiClient.get(`${ENDPOINTS.TEAMS.DETAIL}/${teamId}/standings`)
};

// Leagues API
export const leaguesAPI = {
  getLeagues: (params = {}) => 
    apiClient.get(ENDPOINTS.LEAGUES.BASE, params),
  
  searchLeagues: (query, params = {}) => 
    apiClient.get(ENDPOINTS.LEAGUES.SEARCH, { q: query, ...params }),
  
  getLeaguesByCountry: (country, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.BY_COUNTRY}/${country}`, params),
  
  getLeaguesByConfederation: (confederation, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.BY_CONFEDERATION}/${confederation}`, params),
  
  getLeagueDetail: (leagueId) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.DETAIL}/${leagueId}`),
  
  getLeagueStandings: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.DETAIL}/${leagueId}/standings`, params),
  
  getLeagueMatches: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.DETAIL}/${leagueId}/matches`, params),
  
  getLeagueTeams: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.DETAIL}/${leagueId}/teams`, params),
  
  getLeagueStatistics: (leagueId, params = {}) => 
    apiClient.get(`${ENDPOINTS.LEAGUES.DETAIL}/${leagueId}/statistics`, params)
};

// Predictions API
export const predictionsAPI = {
  getPredictions: (params = {}) => 
    apiClient.get(ENDPOINTS.PREDICTIONS.BASE, params),
  
  getMatchPrediction: (matchId) => 
    apiClient.get(`${ENDPOINTS.PREDICTIONS.BY_MATCH}/${matchId}`),
  
  getPredictionsByDate: (date, params = {}) => 
    apiClient.get(`${ENDPOINTS.PREDICTIONS.BY_DATE}/${date}`, params),
  
  getSimilarMatches: (matchId, params = {}) => 
    apiClient.get(`${ENDPOINTS.PREDICTIONS.SIMILAR_MATCHES}/${matchId}`, params),
  
  getModelPerformance: (params = {}) => 
    apiClient.get(ENDPOINTS.PREDICTIONS.MODEL_PERFORMANCE, params),
  
  getPredictionAccuracy: (params = {}) => 
    apiClient.get(ENDPOINTS.PREDICTIONS.ACCURACY, params),
  
  getPredictionTrends: (params = {}) => 
    apiClient.get(ENDPOINTS.PREDICTIONS.TRENDS, params)
};

// Injuries API
export const injuriesAPI = {
  getInjuries: (params = {}) => 
    apiClient.get(ENDPOINTS.INJURIES.BASE, params),
  
  getActiveInjuries: (params = {}) => 
    apiClient.get(ENDPOINTS.INJURIES.ACTIVE, params),
  
  getPlayerInjuries: (playerId, params = {}) => 
    apiClient.get(`${ENDPOINTS.INJURIES.BY_PLAYER}/${playerId}`, params),
  
  getTeamInjuries: (teamId, params = {}) => 
    apiClient.get(`${ENDPOINTS.INJURIES.BY_TEAM}/${teamId}`, params),
  
  getInjuriesByType: (type, params = {}) => 
    apiClient.get(`${ENDPOINTS.INJURIES.BY_TYPE}/${type}`, params),
  
  getInjuryStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.INJURIES.STATISTICS, params),
  
  getRecoveryUpdates: (params = {}) => 
    apiClient.get(ENDPOINTS.INJURIES.RECOVERY, params),
  
  reportInjury: (injuryData) => 
    apiClient.post(ENDPOINTS.INJURIES.BASE, injuryData),
  
  updateInjury: (injuryId, updateData) => 
    apiClient.put(`${ENDPOINTS.INJURIES.BASE}/${injuryId}`, updateData)
};

// Statistics API
export const statisticsAPI = {
  getGlobalStatistics: () => 
    apiClient.get(ENDPOINTS.STATISTICS.GLOBAL),
  
  getMatchStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.STATISTICS.MATCHES, params),
  
  getPlayerStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.STATISTICS.PLAYERS, params),
  
  getTeamStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.STATISTICS.TEAMS, params),
  
  getLeagueStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.STATISTICS.LEAGUES, params),
  
  getPerformanceStatistics: (params = {}) => 
    apiClient.get(ENDPOINTS.STATISTICS.PERFORMANCE, params)
};

// Reference Data API
export const referenceAPI = {
  getCountries: () => 
    apiClient.get(ENDPOINTS.REFERENCE.COUNTRIES),
  
  getConfederations: () => 
    apiClient.get(ENDPOINTS.REFERENCE.CONFEDERATIONS),
  
  getPositions: () => 
    apiClient.get(ENDPOINTS.REFERENCE.POSITIONS),
  
  getInjuryTypes: () => 
    apiClient.get(ENDPOINTS.REFERENCE.INJURY_TYPES),
  
  getSeasons: () => 
    apiClient.get(ENDPOINTS.REFERENCE.SEASONS)
};

// Utility functions
export const apiUtils = {
  // Set authentication tokens
  setAuthTokens: (accessToken, refreshToken) => {
    apiClient.setAuthTokens(accessToken, refreshToken);
  },

  // Clear authentication tokens
  clearAuthTokens: () => {
    apiClient.clearAuthTokens();
  },

  // Subscribe to auth events
  onAuthEvent: (callback) => {
    return apiClient.onAuthEvent(callback);
  },

  // Cache management
  clearCache: () => {
    apiCache.clear();
  },

  clearExpiredCache: () => {
    apiCache.clearExpired();
  },

  getCacheStats: () => {
    return {
      size: apiCache.cache.size,
      keys: Array.from(apiCache.cache.keys())
    };
  },

  // Error helpers
  isNetworkError: (error) => {
    return error?.code === API_ERRORS.NETWORK_ERROR;
  },

  isAuthError: (error) => {
    return error?.code === API_ERRORS.UNAUTHORIZED || error?.code === API_ERRORS.FORBIDDEN;
  },

  isValidationError: (error) => {
    return error?.code === API_ERRORS.VALIDATION_ERROR;
  },

  // Data formatters
  formatMatchData: (match) => {
    return {
      ...match,
      date: new Date(match.date),
      formattedDate: new Date(match.date).toLocaleDateString(),
      formattedTime: new Date(match.date).toLocaleTimeString(),
      isLive: match.status === 'live',
      isUpcoming: match.status === 'upcoming',
      isFinished: match.status === 'finished'
    };
  },

  formatPlayerData: (player) => {
    return {
      ...player,
      age: player.date_of_birth ? 
        Math.floor((new Date() - new Date(player.date_of_birth)) / (365.25 * 24 * 60 * 60 * 1000)) : 
        null,
      formattedMarketValue: player.market_value ? 
        new Intl.NumberFormat('en-US', { 
          style: 'currency', 
          currency: 'EUR',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        }).format(player.market_value) : 
        'N/A'
    };
  },

  formatTeamData: (team) => {
    return {
      ...team,
      formattedStadiumCapacity: team.stadium_capacity ? 
        new Intl.NumberFormat('en-US').format(team.stadium_capacity) : 
        'N/A'
    };
  },

  // Pagination helper
  buildPaginationParams: (page, limit, sortBy = null, sortOrder = 'asc') => {
    const params = {
      page: Math.max(1, page),
      limit: Math.min(100, Math.max(1, limit))
    };

    if (sortBy) {
      params.sortBy = sortBy;
      params.sortOrder = sortOrder;
    }

    return params;
  }
};

// Export the main API client instance
export default apiClient;

// Export specific APIs
export {
  API_ERRORS, authAPI, ENDPOINTS, favoritesAPI, injuriesAPI, leaguesAPI, matchesAPI,
  playersAPI, predictionsAPI, referenceAPI, statisticsAPI, STATUS_CODES, teamsAPI, userAPI
};
