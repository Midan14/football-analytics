import { createContext, useContext, useEffect, useReducer } from 'react';

// Initial state for the application
const initialState = {
  // User Authentication
  user: null,
  isAuthenticated: false,
  token: null,
  
  // App Settings
  theme: 'light', // 'light' | 'dark' | 'system'
  language: 'en',
  timezone: 'America/Bogota',
  currency: 'EUR',
  
  // Global Loading States
  loading: {
    global: false,
    matches: false,
    players: false,
    teams: false,
    leagues: false,
    predictions: false,
    injuries: false,
    statistics: false
  },
  
  // Error Handling
  errors: {
    global: null,
    api: null,
    network: null
  },
  
  // Favorites Management
  favorites: {
    matches: [],
    players: [],
    teams: [],
    leagues: [],
    predictions: [],
    lastSync: null
  },
  
  // Global Filters
  filters: {
    // Match Filters
    matchStatus: 'all', // 'live' | 'upcoming' | 'finished' | 'all'
    timeframe: 'today', // 'today' | 'tomorrow' | 'week' | 'month'
    confederation: 'all', // 'UEFA' | 'CONMEBOL' | 'CONCACAF' | 'AFC' | 'CAF' | 'OFC' | 'all'
    league: 'all',
    country: 'all',
    
    // Player Filters
    position: 'all', // 'GK' | 'DEF' | 'MID' | 'FWD' | 'all'
    ageRange: { min: 16, max: 45 },
    availability: 'all', // 'available' | 'injured' | 'all'
    marketValueRange: { min: 0, max: 200000000 },
    
    // Team Filters
    division: 'all', // '1st' | '2nd' | '3rd' | 'all'
    foundedRange: { min: 1850, max: new Date().getFullYear() },
    stadiumCapacity: { min: 0, max: 100000 },
    
    // Injury Filters
    injuryStatus: 'all', // 'active' | 'recovered' | 'pending' | 'all'
    severity: 'all', // 'minor' | 'moderate' | 'major' | 'career_ending' | 'all'
    injuryType: 'all'
  },
  
  // Search State
  search: {
    query: '',
    suggestions: [],
    history: [],
    filters: {},
    results: null
  },
  
  // App Configuration
  config: {
    autoRefresh: false,
    refreshInterval: 30000, // 30 seconds
    notifications: true,
    soundEnabled: true,
    compactMode: false,
    showPredictionConfidence: true,
    defaultView: 'cards', // 'cards' | 'table'
    itemsPerPage: 20,
    enableAnalytics: true,
    betaFeatures: false
  },
  
  // Connectivity
  connectivity: {
    isOnline: navigator.onLine || true,
    lastSeen: new Date(),
    syncStatus: 'synced' // 'synced' | 'syncing' | 'offline' | 'error'
  },
  
  // Notifications
  notifications: [],
  
  // Cache Management
  cache: {
    leagues: [],
    countries: [],
    teams: [],
    lastUpdated: null,
    version: '1.0.0'
  },
  
  // Statistics
  stats: {
    totalMatches: 0,
    liveMatches: 0,
    todayMatches: 0,
    predictionAccuracy: 0,
    totalPlayers: 0,
    activePlayers: 0,
    injuredPlayers: 0,
    totalTeams: 0,
    leaguesCovered: 0,
    countriesCovered: 0
  }
};

// Action types
export const ActionTypes = {
  // Authentication
  SET_USER: 'SET_USER',
  SET_AUTHENTICATED: 'SET_AUTHENTICATED',
  LOGOUT: 'LOGOUT',
  
  // App Settings
  SET_THEME: 'SET_THEME',
  SET_LANGUAGE: 'SET_LANGUAGE',
  SET_TIMEZONE: 'SET_TIMEZONE',
  SET_CURRENCY: 'SET_CURRENCY',
  
  // Loading States
  SET_LOADING: 'SET_LOADING',
  SET_GLOBAL_LOADING: 'SET_GLOBAL_LOADING',
  
  // Error Handling
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  CLEAR_ALL_ERRORS: 'CLEAR_ALL_ERRORS',
  
  // Favorites
  SET_FAVORITES: 'SET_FAVORITES',
  ADD_FAVORITE: 'ADD_FAVORITE',
  REMOVE_FAVORITE: 'REMOVE_FAVORITE',
  SYNC_FAVORITES: 'SYNC_FAVORITES',
  
  // Filters
  SET_FILTER: 'SET_FILTER',
  SET_FILTERS: 'SET_FILTERS',
  RESET_FILTERS: 'RESET_FILTERS',
  
  // Search
  SET_SEARCH_QUERY: 'SET_SEARCH_QUERY',
  SET_SEARCH_RESULTS: 'SET_SEARCH_RESULTS',
  ADD_SEARCH_HISTORY: 'ADD_SEARCH_HISTORY',
  CLEAR_SEARCH: 'CLEAR_SEARCH',
  
  // Configuration
  SET_CONFIG: 'SET_CONFIG',
  TOGGLE_CONFIG: 'TOGGLE_CONFIG',
  RESET_CONFIG: 'RESET_CONFIG',
  
  // Connectivity
  SET_ONLINE_STATUS: 'SET_ONLINE_STATUS',
  SET_SYNC_STATUS: 'SET_SYNC_STATUS',
  
  // Notifications
  ADD_NOTIFICATION: 'ADD_NOTIFICATION',
  REMOVE_NOTIFICATION: 'REMOVE_NOTIFICATION',
  CLEAR_NOTIFICATIONS: 'CLEAR_NOTIFICATIONS',
  
  // Cache
  SET_CACHE: 'SET_CACHE',
  UPDATE_CACHE: 'UPDATE_CACHE',
  CLEAR_CACHE: 'CLEAR_CACHE',
  
  // Statistics
  SET_STATS: 'SET_STATS',
  UPDATE_STATS: 'UPDATE_STATS'
};

// Reducer function
const appReducer = (state, action) => {
  switch (action.type) {
    case ActionTypes.SET_USER:
      return {
        ...state,
        user: action.payload,
        isAuthenticated: !!action.payload
      };
      
    case ActionTypes.SET_AUTHENTICATED:
      return {
        ...state,
        isAuthenticated: action.payload,
        token: action.payload ? state.token : null
      };
      
    case ActionTypes.LOGOUT:
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        token: null,
        favorites: initialState.favorites
      };
      
    case ActionTypes.SET_THEME:
      return {
        ...state,
        theme: action.payload
      };
      
    case ActionTypes.SET_LANGUAGE:
      return {
        ...state,
        language: action.payload
      };
      
    case ActionTypes.SET_TIMEZONE:
      return {
        ...state,
        timezone: action.payload
      };
      
    case ActionTypes.SET_CURRENCY:
      return {
        ...state,
        currency: action.payload
      };
      
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: {
          ...state.loading,
          [action.payload.type]: action.payload.value
        }
      };
      
    case ActionTypes.SET_GLOBAL_LOADING:
      return {
        ...state,
        loading: {
          ...state.loading,
          global: action.payload
        }
      };
      
    case ActionTypes.SET_ERROR:
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.payload.type]: action.payload.message
        }
      };
      
    case ActionTypes.CLEAR_ERROR:
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.payload]: null
        }
      };
      
    case ActionTypes.CLEAR_ALL_ERRORS:
      return {
        ...state,
        errors: {
          global: null,
          api: null,
          network: null
        }
      };
      
    case ActionTypes.SET_FAVORITES:
      return {
        ...state,
        favorites: {
          ...action.payload,
          lastSync: new Date()
        }
      };
      
    case ActionTypes.ADD_FAVORITE:
      return {
        ...state,
        favorites: {
          ...state.favorites,
          [action.payload.type]: [
            ...state.favorites[action.payload.type],
            action.payload.item
          ]
        }
      };
      
    case ActionTypes.REMOVE_FAVORITE:
      return {
        ...state,
        favorites: {
          ...state.favorites,
          [action.payload.type]: state.favorites[action.payload.type].filter(
            item => item.id !== action.payload.id
          )
        }
      };
      
    case ActionTypes.SET_FILTER:
      return {
        ...state,
        filters: {
          ...state.filters,
          [action.payload.key]: action.payload.value
        }
      };
      
    case ActionTypes.SET_FILTERS:
      return {
        ...state,
        filters: {
          ...state.filters,
          ...action.payload
        }
      };
      
    case ActionTypes.RESET_FILTERS:
      return {
        ...state,
        filters: initialState.filters
      };
      
    case ActionTypes.SET_SEARCH_QUERY:
      return {
        ...state,
        search: {
          ...state.search,
          query: action.payload
        }
      };
      
    case ActionTypes.SET_SEARCH_RESULTS:
      return {
        ...state,
        search: {
          ...state.search,
          results: action.payload
        }
      };
      
    case ActionTypes.ADD_SEARCH_HISTORY:
      return {
        ...state,
        search: {
          ...state.search,
          history: [
            action.payload,
            ...state.search.history.filter(item => item !== action.payload)
          ].slice(0, 10) // Keep only last 10 searches
        }
      };
      
    case ActionTypes.CLEAR_SEARCH:
      return {
        ...state,
        search: {
          query: '',
          suggestions: [],
          history: state.search.history,
          filters: {},
          results: null
        }
      };
      
    case ActionTypes.SET_CONFIG:
      return {
        ...state,
        config: {
          ...state.config,
          [action.payload.key]: action.payload.value
        }
      };
      
    case ActionTypes.TOGGLE_CONFIG:
      return {
        ...state,
        config: {
          ...state.config,
          [action.payload]: !state.config[action.payload]
        }
      };
      
    case ActionTypes.RESET_CONFIG:
      return {
        ...state,
        config: initialState.config
      };
      
    case ActionTypes.SET_ONLINE_STATUS:
      return {
        ...state,
        connectivity: {
          ...state.connectivity,
          isOnline: action.payload,
          lastSeen: action.payload ? new Date() : state.connectivity.lastSeen
        }
      };
      
    case ActionTypes.SET_SYNC_STATUS:
      return {
        ...state,
        connectivity: {
          ...state.connectivity,
          syncStatus: action.payload
        }
      };
      
    case ActionTypes.ADD_NOTIFICATION:
      return {
        ...state,
        notifications: [
          {
            id: Date.now(),
            timestamp: new Date(),
            ...action.payload
          },
          ...state.notifications
        ]
      };
      
    case ActionTypes.REMOVE_NOTIFICATION:
      return {
        ...state,
        notifications: state.notifications.filter(
          notification => notification.id !== action.payload
        )
      };
      
    case ActionTypes.CLEAR_NOTIFICATIONS:
      return {
        ...state,
        notifications: []
      };
      
    case ActionTypes.SET_CACHE:
      return {
        ...state,
        cache: {
          ...action.payload,
          lastUpdated: new Date()
        }
      };
      
    case ActionTypes.UPDATE_CACHE:
      return {
        ...state,
        cache: {
          ...state.cache,
          [action.payload.key]: action.payload.value,
          lastUpdated: new Date()
        }
      };
      
    case ActionTypes.CLEAR_CACHE:
      return {
        ...state,
        cache: {
          leagues: [],
          countries: [],
          teams: [],
          lastUpdated: null,
          version: '1.0.0'
        }
      };
      
    case ActionTypes.SET_STATS:
      return {
        ...state,
        stats: action.payload
      };
      
    case ActionTypes.UPDATE_STATS:
      return {
        ...state,
        stats: {
          ...state.stats,
          ...action.payload
        }
      };
      
    default:
      return state;
  }
};

// Create context
const AppContext = createContext();

// Context provider component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load persisted state from localStorage on mount
  useEffect(() => {
    try {
      const persistedState = localStorage.getItem('football-analytics-state');
      if (persistedState) {
        const parsed = JSON.parse(persistedState);
        
        // Restore user authentication
        if (parsed.user && parsed.token) {
          dispatch({ type: ActionTypes.SET_USER, payload: parsed.user });
          dispatch({ type: ActionTypes.SET_AUTHENTICATED, payload: true });
        }
        
        // Restore app settings
        if (parsed.theme) {
          dispatch({ type: ActionTypes.SET_THEME, payload: parsed.theme });
        }
        if (parsed.language) {
          dispatch({ type: ActionTypes.SET_LANGUAGE, payload: parsed.language });
        }
        if (parsed.config) {
          Object.entries(parsed.config).forEach(([key, value]) => {
            dispatch({ type: ActionTypes.SET_CONFIG, payload: { key, value } });
          });
        }
        
        // Restore favorites
        if (parsed.favorites) {
          dispatch({ type: ActionTypes.SET_FAVORITES, payload: parsed.favorites });
        }
        
        // Restore cache
        if (parsed.cache) {
          dispatch({ type: ActionTypes.SET_CACHE, payload: parsed.cache });
        }
      }
    } catch (error) {
      console.error('Error loading persisted state:', error);
    }
  }, []);

  // Persist state changes to localStorage
  useEffect(() => {
    try {
      const stateToPersist = {
        user: state.user,
        token: state.token,
        theme: state.theme,
        language: state.language,
        timezone: state.timezone,
        currency: state.currency,
        config: state.config,
        favorites: state.favorites,
        cache: state.cache
      };
      
      localStorage.setItem('football-analytics-state', JSON.stringify(stateToPersist));
    } catch (error) {
      console.error('Error persisting state:', error);
    }
  }, [state.user, state.token, state.theme, state.language, state.timezone, state.currency, state.config, state.favorites, state.cache]);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => {
      dispatch({ type: ActionTypes.SET_ONLINE_STATUS, payload: true });
      dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: 'synced' });
    };
    
    const handleOffline = () => {
      dispatch({ type: ActionTypes.SET_ONLINE_STATUS, payload: false });
      dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: 'offline' });
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (!state.config.autoRefresh || !state.connectivity.isOnline) return;

    const interval = setInterval(() => {
      // Trigger refresh for relevant components
      if (state.loading.matches || state.loading.predictions) {
        // Skip refresh if already loading
        return;
      }
      
      // Dispatch refresh events
      window.dispatchEvent(new CustomEvent('autoRefresh', {
        detail: { type: 'matches' }
      }));
    }, state.config.refreshInterval);

    return () => clearInterval(interval);
  }, [state.config.autoRefresh, state.config.refreshInterval, state.connectivity.isOnline, state.loading.matches, state.loading.predictions]);

  // Action creators
  const actions = {
    // Authentication
    setUser: (user) => dispatch({ type: ActionTypes.SET_USER, payload: user }),
    setAuthenticated: (status) => dispatch({ type: ActionTypes.SET_AUTHENTICATED, payload: status }),
    logout: () => dispatch({ type: ActionTypes.LOGOUT }),
    
    // App Settings
    setTheme: (theme) => dispatch({ type: ActionTypes.SET_THEME, payload: theme }),
    setLanguage: (language) => dispatch({ type: ActionTypes.SET_LANGUAGE, payload: language }),
    setTimezone: (timezone) => dispatch({ type: ActionTypes.SET_TIMEZONE, payload: timezone }),
    setCurrency: (currency) => dispatch({ type: ActionTypes.SET_CURRENCY, payload: currency }),
    
    // Loading States
    setLoading: (type, value) => dispatch({ type: ActionTypes.SET_LOADING, payload: { type, value } }),
    setGlobalLoading: (value) => dispatch({ type: ActionTypes.SET_GLOBAL_LOADING, payload: value }),
    
    // Error Handling
    setError: (type, message) => dispatch({ type: ActionTypes.SET_ERROR, payload: { type, message } }),
    clearError: (type) => dispatch({ type: ActionTypes.CLEAR_ERROR, payload: type }),
    clearAllErrors: () => dispatch({ type: ActionTypes.CLEAR_ALL_ERRORS }),
    
    // Favorites
    setFavorites: (favorites) => dispatch({ type: ActionTypes.SET_FAVORITES, payload: favorites }),
    addFavorite: (type, item) => dispatch({ type: ActionTypes.ADD_FAVORITE, payload: { type, item } }),
    removeFavorite: (type, id) => dispatch({ type: ActionTypes.REMOVE_FAVORITE, payload: { type, id } }),
    
    // Filters
    setFilter: (key, value) => dispatch({ type: ActionTypes.SET_FILTER, payload: { key, value } }),
    setFilters: (filters) => dispatch({ type: ActionTypes.SET_FILTERS, payload: filters }),
    resetFilters: () => dispatch({ type: ActionTypes.RESET_FILTERS }),
    
    // Search
    setSearchQuery: (query) => dispatch({ type: ActionTypes.SET_SEARCH_QUERY, payload: query }),
    setSearchResults: (results) => dispatch({ type: ActionTypes.SET_SEARCH_RESULTS, payload: results }),
    addSearchHistory: (query) => dispatch({ type: ActionTypes.ADD_SEARCH_HISTORY, payload: query }),
    clearSearch: () => dispatch({ type: ActionTypes.CLEAR_SEARCH }),
    
    // Configuration
    setConfig: (key, value) => dispatch({ type: ActionTypes.SET_CONFIG, payload: { key, value } }),
    toggleConfig: (key) => dispatch({ type: ActionTypes.TOGGLE_CONFIG, payload: key }),
    resetConfig: () => dispatch({ type: ActionTypes.RESET_CONFIG }),
    
    // Connectivity
    setOnlineStatus: (status) => dispatch({ type: ActionTypes.SET_ONLINE_STATUS, payload: status }),
    setSyncStatus: (status) => dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: status }),
    
    // Notifications
    addNotification: (notification) => dispatch({ type: ActionTypes.ADD_NOTIFICATION, payload: notification }),
    removeNotification: (id) => dispatch({ type: ActionTypes.REMOVE_NOTIFICATION, payload: id }),
    clearNotifications: () => dispatch({ type: ActionTypes.CLEAR_NOTIFICATIONS }),
    
    // Cache
    setCache: (cache) => dispatch({ type: ActionTypes.SET_CACHE, payload: cache }),
    updateCache: (key, value) => dispatch({ type: ActionTypes.UPDATE_CACHE, payload: { key, value } }),
    clearCache: () => dispatch({ type: ActionTypes.CLEAR_CACHE }),
    
    // Statistics
    setStats: (stats) => dispatch({ type: ActionTypes.SET_STATS, payload: stats }),
    updateStats: (stats) => dispatch({ type: ActionTypes.UPDATE_STATS, payload: stats })
  };

  // Helper functions
  const helpers = {
    // Check if item is in favorites
    isFavorite: (type, id) => {
      return state.favorites[type]?.some(item => item.id === id) || false;
    },
    
    // Toggle favorite
    toggleFavorite: async (type, item) => {
      const isFav = helpers.isFavorite(type, item.id);
      
      try {
        if (isFav) {
          actions.removeFavorite(type, item.id);
          // API call to remove from backend
          if (state.isAuthenticated) {
            await fetch(`/api/user/favorites/${type}/${item.id}`, {
              method: 'DELETE',
              headers: {
                'Authorization': `Bearer ${state.token}`
              }
            });
          }
        } else {
          actions.addFavorite(type, item);
          // API call to add to backend
          if (state.isAuthenticated) {
            await fetch(`/api/user/favorites/${type}/${item.id}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${state.token}`
              },
              body: JSON.stringify(item)
            });
          }
        }
      } catch (error) {
        console.error('Error toggling favorite:', error);
        actions.setError('api', 'Failed to update favorites');
      }
    },
    
    // Format currency
    formatCurrency: (amount) => {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: state.currency,
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
      }).format(amount);
    },
    
    // Format date based on timezone
    formatDate: (date, options = {}) => {
      return new Intl.DateTimeFormat(state.language, {
        timeZone: state.timezone,
        ...options
      }).format(new Date(date));
    },
    
    // Get filtered data based on current filters
    getFilteredData: (data, type) => {
      if (!data || !Array.isArray(data)) return [];
      
      return data.filter(item => {
        // Apply filters based on type
        switch (type) {
          case 'matches':
            if (state.filters.matchStatus !== 'all' && item.status !== state.filters.matchStatus) return false;
            if (state.filters.league !== 'all' && item.league_id !== state.filters.league) return false;
            if (state.filters.confederation !== 'all' && item.confederation !== state.filters.confederation) return false;
            break;
            
          case 'players':
            if (state.filters.position !== 'all' && item.position !== state.filters.position) return false;
            if (item.age < state.filters.ageRange.min || item.age > state.filters.ageRange.max) return false;
            if (state.filters.availability !== 'all') {
              const isInjured = item.injury_status === 'active';
              if (state.filters.availability === 'available' && isInjured) return false;
              if (state.filters.availability === 'injured' && !isInjured) return false;
            }
            break;
            
          case 'teams':
            if (state.filters.league !== 'all' && item.league_id !== state.filters.league) return false;
            if (state.filters.country !== 'all' && item.country !== state.filters.country) return false;
            if (state.filters.division !== 'all' && item.division !== state.filters.division) return false;
            break;
            
          default:
            break;
        }
        
        return true;
      });
    }
  };

  const value = {
    state,
    actions,
    helpers,
    dispatch
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

// Export context for advanced usage
export default AppContext;