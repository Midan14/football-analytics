import { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';

// Storage configuration
const STORAGE_CONFIG = {
  prefix: 'football-analytics-',
  version: '1.0.0',
  compression: {
    enabled: true,
    threshold: 1024, // Compress data larger than 1KB
  },
  encryption: {
    enabled: false, // Can be enabled for sensitive data
    key: null
  },
  sync: {
    enabled: true, // Sync between tabs
    debounceTime: 100
  }
};

// Storage keys used throughout the application
export const STORAGE_KEYS = {
  // Authentication
  AUTH_TOKEN: 'auth-token',
  REFRESH_TOKEN: 'refresh-token',
  USER_DATA: 'user-data',
  SESSION_DATA: 'session-data',
  
  // App Settings
  THEME: 'theme',
  LANGUAGE: 'language',
  TIMEZONE: 'timezone',
  CURRENCY: 'currency',
  
  // User Preferences
  PREFERENCES: 'preferences',
  FAVORITES: 'favorites',
  FILTERS: 'filters',
  SEARCH_HISTORY: 'search-history',
  
  // App Configuration
  CONFIG: 'config',
  LAYOUT_SETTINGS: 'layout-settings',
  NOTIFICATION_SETTINGS: 'notification-settings',
  
  // Cache
  API_CACHE: 'api-cache',
  LEAGUES_CACHE: 'leagues-cache',
  TEAMS_CACHE: 'teams-cache',
  COUNTRIES_CACHE: 'countries-cache',
  
  // Analytics & Usage
  ANALYTICS: 'analytics',
  FEATURE_USAGE: 'feature-usage',
  PERFORMANCE_METRICS: 'performance-metrics',
  
  // Temporary Data
  FORM_DRAFTS: 'form-drafts',
  PREDICTIONS_CACHE: 'predictions-cache',
  STATISTICS_CACHE: 'statistics-cache'
};

// Data validation schemas
const VALIDATION_SCHEMAS = {
  [STORAGE_KEYS.USER_DATA]: (data) => {
    return data && typeof data === 'object' && data.id && data.email;
  },
  [STORAGE_KEYS.FAVORITES]: (data) => {
    return data && typeof data === 'object' && 
           Array.isArray(data.matches) && Array.isArray(data.players) &&
           Array.isArray(data.teams) && Array.isArray(data.leagues);
  },
  [STORAGE_KEYS.FILTERS]: (data) => {
    return data && typeof data === 'object';
  },
  [STORAGE_KEYS.CONFIG]: (data) => {
    return data && typeof data === 'object' && 
           typeof data.autoRefresh === 'boolean';
  }
};

// TTL (Time To Live) settings for different data types
const TTL_SETTINGS = {
  [STORAGE_KEYS.API_CACHE]: 5 * 60 * 1000, // 5 minutes
  [STORAGE_KEYS.LEAGUES_CACHE]: 24 * 60 * 60 * 1000, // 24 hours
  [STORAGE_KEYS.TEAMS_CACHE]: 12 * 60 * 60 * 1000, // 12 hours
  [STORAGE_KEYS.COUNTRIES_CACHE]: 7 * 24 * 60 * 60 * 1000, // 7 days
  [STORAGE_KEYS.PREDICTIONS_CACHE]: 30 * 60 * 1000, // 30 minutes
  [STORAGE_KEYS.STATISTICS_CACHE]: 60 * 60 * 1000, // 1 hour
  [STORAGE_KEYS.FORM_DRAFTS]: 24 * 60 * 60 * 1000, // 24 hours
};

// Compression utilities
const compress = (data) => {
  if (!STORAGE_CONFIG.compression.enabled) return data;
  
  try {
    const jsonString = JSON.stringify(data);
    if (jsonString.length < STORAGE_CONFIG.compression.threshold) {
      return data;
    }
    
    // Simple compression using btoa (base64)
    // In production, you might want to use a more sophisticated compression library
    return {
      __compressed: true,
      data: btoa(jsonString)
    };
  } catch (error) {
    console.warn('Compression failed:', error);
    return data;
  }
};

const decompress = (data) => {
  if (!data || typeof data !== 'object' || !data.__compressed) {
    return data;
  }
  
  try {
    const jsonString = atob(data.data);
    return JSON.parse(jsonString);
  } catch (error) {
    console.warn('Decompression failed:', error);
    return null;
  }
};

// Storage wrapper with metadata
const createStorageItem = (value, options = {}) => {
  const now = Date.now();
  const ttl = options.ttl || TTL_SETTINGS[options.key];
  
  return {
    value: compress(value),
    timestamp: now,
    expiresAt: ttl ? now + ttl : null,
    version: STORAGE_CONFIG.version,
    metadata: {
      userAgent: navigator.userAgent,
      url: window.location.href,
      ...options.metadata
    }
  };
};

const parseStorageItem = (item) => {
  if (!item) return null;
  
  try {
    const parsed = JSON.parse(item);
    
    // Check expiration
    if (parsed.expiresAt && Date.now() > parsed.expiresAt) {
      return null;
    }
    
    // Check version compatibility
    if (parsed.version !== STORAGE_CONFIG.version) {
      console.warn('Storage version mismatch, clearing item');
      return null;
    }
    
    return {
      ...parsed,
      value: decompress(parsed.value)
    };
  } catch (error) {
    console.warn('Failed to parse storage item:', error);
    return null;
  }
};

// Main useLocalStorage hook
export const useLocalStorage = (key, defaultValue = null, options = {}) => {
  const { actions: appActions } = useApp();
  const fullKey = `${STORAGE_CONFIG.prefix}${key}`;
  
  // Options with defaults
  const config = {
    serialize: JSON.stringify,
    deserialize: JSON.parse,
    validator: VALIDATION_SCHEMAS[key],
    syncAcrossTabs: STORAGE_CONFIG.sync.enabled,
    ttl: TTL_SETTINGS[key],
    onError: (error) => console.error('LocalStorage error:', error),
    onSync: null, // Callback when synced from another tab
    ...options
  };
  
  // State
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = localStorage.getItem(fullKey);
      const parsed = parseStorageItem(item);
      
      if (parsed === null) {
        return defaultValue;
      }
      
      // Validate if validator exists
      if (config.validator && !config.validator(parsed.value)) {
        console.warn(`Invalid data for key ${key}, using default value`);
        return defaultValue;
      }
      
      return parsed.value;
    } catch (error) {
      config.onError(error);
      return defaultValue;
    }
  });
  
  // Refs for cleanup and control
  const syncTimeoutRef = useRef(null);
  const isSettingRef = useRef(false);
  
  // Set value function
  const setValue = useCallback((value) => {
    try {
      isSettingRef.current = true;
      
      // Allow value to be a function for functional updates
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      
      // Validate if validator exists
      if (config.validator && !config.validator(valueToStore)) {
        throw new Error('Validation failed for storage value');
      }
      
      // Create storage item with metadata
      const storageItem = createStorageItem(valueToStore, {
        key,
        ttl: config.ttl,
        metadata: options.metadata
      });
      
      // Store in localStorage
      localStorage.setItem(fullKey, JSON.stringify(storageItem));
      
      // Update state
      setStoredValue(valueToStore);
      
      // Trigger sync notification
      if (config.syncAcrossTabs) {
        window.dispatchEvent(new CustomEvent('localStorage-change', {
          detail: { key: fullKey, value: valueToStore }
        }));
      }
      
    } catch (error) {
      config.onError(error);
    } finally {
      isSettingRef.current = false;
    }
  }, [storedValue, fullKey, key, config]);
  
  // Remove value function
  const removeValue = useCallback(() => {
    try {
      localStorage.removeItem(fullKey);
      setStoredValue(defaultValue);
      
      if (config.syncAcrossTabs) {
        window.dispatchEvent(new CustomEvent('localStorage-change', {
          detail: { key: fullKey, value: null }
        }));
      }
    } catch (error) {
      config.onError(error);
    }
  }, [fullKey, defaultValue, config]);
  
  // Check if value exists
  const exists = useCallback(() => {
    try {
      const item = localStorage.getItem(fullKey);
      const parsed = parseStorageItem(item);
      return parsed !== null;
    } catch (error) {
      return false;
    }
  }, [fullKey]);
  
  // Get metadata
  const getMetadata = useCallback(() => {
    try {
      const item = localStorage.getItem(fullKey);
      const parsed = parseStorageItem(item);
      return parsed?.metadata || null;
    } catch (error) {
      return null;
    }
  }, [fullKey]);
  
  // Update metadata without changing value
  const updateMetadata = useCallback((metadata) => {
    setValue(storedValue); // This will update metadata through createStorageItem
  }, [setValue, storedValue]);
  
  // Listen for storage changes from other tabs
  useEffect(() => {
    if (!config.syncAcrossTabs) return;
    
    const handleStorageChange = (e) => {
      if (e.detail.key === fullKey && !isSettingRef.current) {
        if (syncTimeoutRef.current) {
          clearTimeout(syncTimeoutRef.current);
        }
        
        syncTimeoutRef.current = setTimeout(() => {
          const newValue = e.detail.value;
          setStoredValue(newValue === null ? defaultValue : newValue);
          
          if (config.onSync) {
            config.onSync(newValue);
          }
        }, STORAGE_CONFIG.sync.debounceTime);
      }
    };
    
    window.addEventListener('localStorage-change', handleStorageChange);
    
    return () => {
      window.removeEventListener('localStorage-change', handleStorageChange);
      if (syncTimeoutRef.current) {
        clearTimeout(syncTimeoutRef.current);
      }
    };
  }, [fullKey, defaultValue, config]);
  
  // Native storage event listener (for cross-tab sync)
  useEffect(() => {
    if (!config.syncAcrossTabs) return;
    
    const handleNativeStorageChange = (e) => {
      if (e.key === fullKey && e.newValue !== e.oldValue) {
        try {
          const parsed = parseStorageItem(e.newValue);
          const newValue = parsed ? parsed.value : defaultValue;
          
          if (!isSettingRef.current) {
            setStoredValue(newValue);
            
            if (config.onSync) {
              config.onSync(newValue);
            }
          }
        } catch (error) {
          config.onError(error);
        }
      }
    };
    
    window.addEventListener('storage', handleNativeStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleNativeStorageChange);
    };
  }, [fullKey, defaultValue, config]);
  
  return {
    value: storedValue,
    setValue,
    removeValue,
    exists,
    getMetadata,
    updateMetadata
  };
};

// Specialized hooks for different data types
export const useAuthStorage = () => {
  const token = useLocalStorage(STORAGE_KEYS.AUTH_TOKEN, null);
  const refreshToken = useLocalStorage(STORAGE_KEYS.REFRESH_TOKEN, null);
  const userData = useLocalStorage(STORAGE_KEYS.USER_DATA, null);
  const sessionData = useLocalStorage(STORAGE_KEYS.SESSION_DATA, null);
  
  const clearAuth = useCallback(() => {
    token.removeValue();
    refreshToken.removeValue();
    userData.removeValue();
    sessionData.removeValue();
  }, [token, refreshToken, userData, sessionData]);
  
  return {
    token: token.value,
    setToken: token.setValue,
    refreshToken: refreshToken.value,
    setRefreshToken: refreshToken.setValue,
    userData: userData.value,
    setUserData: userData.setValue,
    sessionData: sessionData.value,
    setSessionData: sessionData.setValue,
    clearAuth
  };
};

export const usePreferencesStorage = () => {
  const defaultPreferences = {
    theme: 'light',
    language: 'en',
    timezone: 'America/Bogota',
    currency: 'EUR',
    defaultView: 'cards',
    itemsPerPage: 20,
    autoRefresh: false,
    notifications: true
  };
  
  return useLocalStorage(STORAGE_KEYS.PREFERENCES, defaultPreferences);
};

export const useFavoritesStorage = () => {
  const defaultFavorites = {
    matches: [],
    players: [],
    teams: [],
    leagues: [],
    predictions: [],
    lastSync: null
  };
  
  const { value: favorites, setValue, ...rest } = useLocalStorage(
    STORAGE_KEYS.FAVORITES, 
    defaultFavorites
  );
  
  const addFavorite = useCallback((type, item) => {
    setValue(prev => ({
      ...prev,
      [type]: [...(prev[type] || []), item],
      lastSync: new Date().toISOString()
    }));
  }, [setValue]);
  
  const removeFavorite = useCallback((type, id) => {
    setValue(prev => ({
      ...prev,
      [type]: (prev[type] || []).filter(item => item.id !== id),
      lastSync: new Date().toISOString()
    }));
  }, [setValue]);
  
  const isFavorite = useCallback((type, id) => {
    return (favorites[type] || []).some(item => item.id === id);
  }, [favorites]);
  
  const clearFavorites = useCallback((type = null) => {
    if (type) {
      setValue(prev => ({
        ...prev,
        [type]: [],
        lastSync: new Date().toISOString()
      }));
    } else {
      setValue(defaultFavorites);
    }
  }, [setValue, defaultFavorites]);
  
  return {
    favorites,
    setFavorites: setValue,
    addFavorite,
    removeFavorite,
    isFavorite,
    clearFavorites,
    ...rest
  };
};

export const useFiltersStorage = () => {
  const defaultFilters = {
    matchStatus: 'all',
    timeframe: 'today',
    confederation: 'all',
    league: 'all',
    country: 'all',
    position: 'all',
    ageRange: { min: 16, max: 45 },
    availability: 'all',
    marketValueRange: { min: 0, max: 200000000 },
    division: 'all',
    foundedRange: { min: 1850, max: new Date().getFullYear() },
    stadiumCapacity: { min: 0, max: 100000 },
    injuryStatus: 'all',
    severity: 'all',
    injuryType: 'all'
  };
  
  const { value: filters, setValue, ...rest } = useLocalStorage(
    STORAGE_KEYS.FILTERS, 
    defaultFilters
  );
  
  const setFilter = useCallback((key, value) => {
    setValue(prev => ({
      ...prev,
      [key]: value
    }));
  }, [setValue]);
  
  const resetFilters = useCallback(() => {
    setValue(defaultFilters);
  }, [setValue, defaultFilters]);
  
  const getActiveFiltersCount = useCallback(() => {
    return Object.entries(filters).filter(([key, value]) => {
      const defaultValue = defaultFilters[key];
      return JSON.stringify(value) !== JSON.stringify(defaultValue);
    }).length;
  }, [filters, defaultFilters]);
  
  return {
    filters,
    setFilters: setValue,
    setFilter,
    resetFilters,
    getActiveFiltersCount,
    ...rest
  };
};

export const useSearchHistoryStorage = () => {
  const { value: history, setValue, ...rest } = useLocalStorage(
    STORAGE_KEYS.SEARCH_HISTORY, 
    []
  );
  
  const addSearch = useCallback((query) => {
    if (!query.trim()) return;
    
    setValue(prev => {
      const filtered = prev.filter(item => item.query !== query);
      return [
        { query, timestamp: new Date().toISOString() },
        ...filtered
      ].slice(0, 50); // Keep only last 50 searches
    });
  }, [setValue]);
  
  const removeSearch = useCallback((query) => {
    setValue(prev => prev.filter(item => item.query !== query));
  }, [setValue]);
  
  const clearHistory = useCallback(() => {
    setValue([]);
  }, [setValue]);
  
  const getRecentSearches = useCallback((limit = 10) => {
    return history.slice(0, limit);
  }, [history]);
  
  return {
    history,
    addSearch,
    removeSearch,
    clearHistory,
    getRecentSearches,
    ...rest
  };
};

export const useCacheStorage = (cacheKey, ttl = null) => {
  const fullKey = `${STORAGE_KEYS.API_CACHE}-${cacheKey}`;
  
  const { value, setValue, removeValue, exists } = useLocalStorage(
    fullKey, 
    null,
    { ttl }
  );
  
  const setCache = useCallback((data, customTtl = null) => {
    setValue(data);
  }, [setValue]);
  
  const clearCache = useCallback(() => {
    removeValue();
  }, [removeValue]);
  
  const isExpired = useCallback(() => {
    if (!exists()) return true;
    
    try {
      const item = localStorage.getItem(`${STORAGE_CONFIG.prefix}${fullKey}`);
      const parsed = parseStorageItem(item);
      return parsed === null;
    } catch {
      return true;
    }
  }, [exists, fullKey]);
  
  return {
    data: value,
    setCache,
    clearCache,
    exists,
    isExpired
  };
};

// Utility functions
export const clearAllStorage = () => {
  try {
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith(STORAGE_CONFIG.prefix)) {
        localStorage.removeItem(key);
      }
    });
  } catch (error) {
    console.error('Failed to clear storage:', error);
  }
};

export const getStorageUsage = () => {
  try {
    let totalSize = 0;
    const usage = {};
    
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith(STORAGE_CONFIG.prefix)) {
        const value = localStorage.getItem(key);
        const size = new Blob([value]).size;
        totalSize += size;
        
        const shortKey = key.replace(STORAGE_CONFIG.prefix, '');
        usage[shortKey] = {
          size,
          formattedSize: formatBytes(size)
        };
      }
    });
    
    return {
      total: totalSize,
      formattedTotal: formatBytes(totalSize),
      items: usage
    };
  } catch (error) {
    console.error('Failed to calculate storage usage:', error);
    return { total: 0, formattedTotal: '0 B', items: {} };
  }
};

export const cleanExpiredStorage = () => {
  try {
    let cleanedCount = 0;
    
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith(STORAGE_CONFIG.prefix)) {
        const item = localStorage.getItem(key);
        const parsed = parseStorageItem(item);
        
        if (parsed === null) {
          localStorage.removeItem(key);
          cleanedCount++;
        }
      }
    });
    
    return cleanedCount;
  } catch (error) {
    console.error('Failed to clean expired storage:', error);
    return 0;
  }
};

// Helper function to format bytes
const formatBytes = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

export default useLocalStorage;