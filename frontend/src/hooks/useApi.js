import { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import { useAuth } from '../context/AuthContext';

// API Configuration
const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000, // 1 second
  cacheTimeout: 5 * 60 * 1000, // 5 minutes
};

// Cache storage
const apiCache = new Map();

// Request queue for preventing duplicate requests
const requestQueue = new Map();

// Custom hook for API interactions
export const useApi = (initialUrl = null, options = {}) => {
  const { getAuthHeaders, refreshAccessToken, isAuthenticated } = useAuth();
  const { actions: appActions, state: appState } = useApp();
  
  // State management
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastFetched, setLastFetched] = useState(null);
  
  // Refs for cleanup and control
  const abortControllerRef = useRef(null);
  const mountedRef = useRef(true);
  
  // Default options
  const defaultOptions = {
    method: 'GET',
    cache: true,
    retry: true,
    loadingType: 'global', // 'global' | 'local' | 'none'
    errorType: 'global', // 'global' | 'local' | 'none'
    timeout: API_CONFIG.timeout,
    retryAttempts: API_CONFIG.retryAttempts,
    retryDelay: API_CONFIG.retryDelay,
    transform: null, // Function to transform response data
    onSuccess: null, // Callback on successful response
    onError: null, // Callback on error
    dependencies: [], // Dependencies for auto-refetch
    ...options
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Auto-refresh effect based on dependencies
  useEffect(() => {
    if (initialUrl && defaultOptions.dependencies.length > 0) {
      executeRequest(initialUrl, defaultOptions);
    }
  }, defaultOptions.dependencies);

  // Generate cache key
  const getCacheKey = useCallback((url, options) => {
    const sortedParams = new URLSearchParams();
    if (options.params) {
      Object.keys(options.params).sort().forEach(key => {
        sortedParams.append(key, options.params[key]);
      });
    }
    return `${options.method || 'GET'}:${url}?${sortedParams.toString()}`;
  }, []);

  // Check if request is cached and valid
  const getCachedData = useCallback((cacheKey) => {
    const cached = apiCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < API_CONFIG.cacheTimeout) {
      return cached.data;
    }
    return null;
  }, []);

  // Set cache data
  const setCacheData = useCallback((cacheKey, data) => {
    apiCache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });
  }, []);

  // Build full URL with parameters
  const buildUrl = useCallback((url, params) => {
    if (!params) return url;
    
    const urlObj = new URL(url.startsWith('http') ? url : `${API_CONFIG.baseURL}${url}`);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        urlObj.searchParams.append(key, value);
      }
    });
    
    return urlObj.toString();
  }, []);

  // Handle loading states
  const setLoadingState = useCallback((isLoading, type) => {
    setLoading(isLoading);
    
    if (type === 'global') {
      appActions.setGlobalLoading(isLoading);
    } else if (type && type !== 'none') {
      appActions.setLoading(type, isLoading);
    }
  }, [appActions]);

  // Handle error states
  const setErrorState = useCallback((error, type) => {
    setError(error);
    
    if (type === 'global') {
      appActions.setError('global', error?.message || 'An error occurred');
    } else if (type && type !== 'none') {
      appActions.setError(type, error?.message || 'An error occurred');
    }
  }, [appActions]);

  // Execute HTTP request with retry logic
  const executeRequest = useCallback(async (url, requestOptions = {}) => {
    const options = { ...defaultOptions, ...requestOptions };
    const fullUrl = buildUrl(url, options.params);
    const cacheKey = getCacheKey(fullUrl, options);

    // Check cache first for GET requests
    if (options.method === 'GET' && options.cache) {
      const cachedData = getCachedData(cacheKey);
      if (cachedData) {
        setData(cachedData);
        setLastFetched(new Date());
        return { success: true, data: cachedData };
      }
    }

    // Check if same request is already in progress
    if (requestQueue.has(cacheKey)) {
      return requestQueue.get(cacheKey);
    }

    // Create abort controller
    abortControllerRef.current = new AbortController();

    // Set loading state
    setLoadingState(true, options.loadingType);
    setErrorState(null, options.errorType);

    const requestPromise = (async () => {
      let lastError = null;
      
      for (let attempt = 0; attempt <= options.retryAttempts; attempt++) {
        try {
          // Prepare request headers
          const headers = {
            'Content-Type': 'application/json',
            ...options.headers
          };

          // Add auth headers if authenticated
          if (isAuthenticated) {
            const authHeaders = getAuthHeaders();
            Object.assign(headers, authHeaders);
          }

          // Prepare request config
          const requestConfig = {
            method: options.method,
            headers,
            signal: abortControllerRef.current.signal,
            ...options.fetchOptions
          };

          // Add body for non-GET requests
          if (options.method !== 'GET' && options.data) {
            requestConfig.body = JSON.stringify(options.data);
          }

          // Create timeout promise
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), options.timeout);
          });

          // Execute request with timeout
          const response = await Promise.race([
            fetch(fullUrl, requestConfig),
            timeoutPromise
          ]);

          // Handle HTTP errors
          if (!response.ok) {
            // Handle 401 Unauthorized - try to refresh token
            if (response.status === 401 && isAuthenticated && attempt === 0) {
              const newToken = await refreshAccessToken();
              if (newToken) {
                // Retry with new token
                continue;
              }
            }

            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
          }

          // Parse response
          let responseData;
          const contentType = response.headers.get('content-type');
          
          if (contentType && contentType.includes('application/json')) {
            responseData = await response.json();
          } else {
            responseData = await response.text();
          }

          // Transform data if transformer provided
          if (options.transform && typeof options.transform === 'function') {
            responseData = options.transform(responseData);
          }

          // Cache successful GET requests
          if (options.method === 'GET' && options.cache) {
            setCacheData(cacheKey, responseData);
          }

          // Update state if component is still mounted
          if (mountedRef.current) {
            setData(responseData);
            setLastFetched(new Date());
            setLoadingState(false, options.loadingType);
            setErrorState(null, options.errorType);
          }

          // Call success callback
          if (options.onSuccess) {
            options.onSuccess(responseData, response);
          }

          return { success: true, data: responseData, response };

        } catch (error) {
          lastError = error;

          // Don't retry on abort or certain errors
          if (error.name === 'AbortError' || 
              error.message.includes('401') || 
              error.message.includes('403') ||
              attempt === options.retryAttempts) {
            break;
          }

          // Wait before retry
          if (attempt < options.retryAttempts) {
            await new Promise(resolve => 
              setTimeout(resolve, options.retryDelay * Math.pow(2, attempt))
            );
          }
        }
      }

      // Handle final error
      if (mountedRef.current) {
        setLoadingState(false, options.loadingType);
        setErrorState(lastError, options.errorType);
      }

      // Call error callback
      if (options.onError) {
        options.onError(lastError);
      }

      return { success: false, error: lastError };
    })();

    // Add to request queue
    requestQueue.set(cacheKey, requestPromise);

    try {
      const result = await requestPromise;
      return result;
    } finally {
      // Remove from queue
      requestQueue.delete(cacheKey);
    }
  }, [
    defaultOptions, buildUrl, getCacheKey, getCachedData, setCacheData,
    setLoadingState, setErrorState, isAuthenticated, getAuthHeaders, refreshAccessToken
  ]);

  // Main request function
  const request = useCallback((url, options = {}) => {
    return executeRequest(url, options);
  }, [executeRequest]);

  // GET request helper
  const get = useCallback((url, params = null, options = {}) => {
    return request(url, { 
      method: 'GET', 
      params, 
      ...options 
    });
  }, [request]);

  // POST request helper
  const post = useCallback((url, data = null, options = {}) => {
    return request(url, { 
      method: 'POST', 
      data, 
      cache: false,
      ...options 
    });
  }, [request]);

  // PUT request helper
  const put = useCallback((url, data = null, options = {}) => {
    return request(url, { 
      method: 'PUT', 
      data, 
      cache: false,
      ...options 
    });
  }, [request]);

  // DELETE request helper
  const del = useCallback((url, options = {}) => {
    return request(url, { 
      method: 'DELETE', 
      cache: false,
      ...options 
    });
  }, [request]);

  // PATCH request helper
  const patch = useCallback((url, data = null, options = {}) => {
    return request(url, { 
      method: 'PATCH', 
      data, 
      cache: false,
      ...options 
    });
  }, [request]);

  // Refetch current data
  const refetch = useCallback(() => {
    if (initialUrl) {
      return executeRequest(initialUrl, defaultOptions);
    }
    return Promise.resolve({ success: false, error: 'No URL to refetch' });
  }, [initialUrl, defaultOptions, executeRequest]);

  // Cancel current request
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  // Clear cache
  const clearCache = useCallback((pattern = null) => {
    if (pattern) {
      // Clear cache entries matching pattern
      Array.from(apiCache.keys()).forEach(key => {
        if (key.includes(pattern)) {
          apiCache.delete(key);
        }
      });
    } else {
      // Clear all cache
      apiCache.clear();
    }
  }, []);

  // Auto-fetch initial data
  useEffect(() => {
    if (initialUrl) {
      executeRequest(initialUrl, defaultOptions);
    }
  }, [initialUrl]);

  return {
    // State
    data,
    loading,
    error,
    lastFetched,
    
    // HTTP Methods
    request,
    get,
    post,
    put,
    patch,
    delete: del,
    
    // Control
    refetch,
    cancel,
    clearCache,
    
    // Utilities
    isOnline: appState.connectivity.isOnline,
    isCached: (url, params) => {
      const cacheKey = getCacheKey(buildUrl(url, params), { method: 'GET', params });
      return !!getCachedData(cacheKey);
    }
  };
};

// Specialized hooks for different API endpoints
export const useMatchesApi = (options = {}) => {
  return useApi('/matches', {
    loadingType: 'matches',
    errorType: 'api',
    ...options
  });
};

export const usePlayersApi = (options = {}) => {
  return useApi('/players', {
    loadingType: 'players',
    errorType: 'api',
    ...options
  });
};

export const useTeamsApi = (options = {}) => {
  return useApi('/teams', {
    loadingType: 'teams',
    errorType: 'api',
    ...options
  });
};

export const useLeaguesApi = (options = {}) => {
  return useApi('/leagues', {
    loadingType: 'leagues',
    errorType: 'api',
    ...options
  });
};

export const usePredictionsApi = (options = {}) => {
  return useApi('/predictions', {
    loadingType: 'predictions',
    errorType: 'api',
    ...options
  });
};

export const useInjuriesApi = (options = {}) => {
  return useApi('/injuries', {
    loadingType: 'injuries',
    errorType: 'api',
    ...options
  });
};

export const useStatisticsApi = (options = {}) => {
  return useApi('/statistics', {
    loadingType: 'statistics',
    errorType: 'api',
    ...options
  });
};

export const useFavoritesApi = (type, options = {}) => {
  return useApi(`/user/favorites/${type}`, {
    loadingType: 'local',
    errorType: 'local',
    ...options
  });
};

// Helper hook for paginated data
export const usePaginatedApi = (url, initialParams = {}, options = {}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialParams.limit || 20);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const params = {
    page: currentPage,
    limit: pageSize,
    ...initialParams
  };

  const api = useApi(url, {
    params,
    transform: (data) => {
      if (data.pagination) {
        setTotalItems(data.pagination.total);
        setTotalPages(data.pagination.pages);
      }
      return data.data || data;
    },
    dependencies: [currentPage, pageSize, JSON.stringify(initialParams)],
    ...options
  });

  const goToPage = useCallback((page) => {
    setCurrentPage(page);
  }, []);

  const changePageSize = useCallback((size) => {
    setPageSize(size);
    setCurrentPage(1);
  }, []);

  const nextPage = useCallback(() => {
    if (currentPage < totalPages) {
      setCurrentPage(prev => prev + 1);
    }
  }, [currentPage, totalPages]);

  const prevPage = useCallback(() => {
    if (currentPage > 1) {
      setCurrentPage(prev => prev - 1);
    }
  }, [currentPage]);

  return {
    ...api,
    pagination: {
      currentPage,
      pageSize,
      totalItems,
      totalPages,
      hasNext: currentPage < totalPages,
      hasPrev: currentPage > 1,
      goToPage,
      nextPage,
      prevPage,
      changePageSize
    }
  };
};

// Helper hook for real-time data
export const useRealtimeApi = (url, interval = 30000, options = {}) => {
  const api = useApi(url, options);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (interval > 0) {
      intervalRef.current = setInterval(() => {
        api.refetch();
      }, interval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [interval, api.refetch]);

  const startPolling = useCallback((newInterval = interval) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    intervalRef.current = setInterval(() => {
      api.refetch();
    }, newInterval);
  }, [interval, api.refetch]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  return {
    ...api,
    startPolling,
    stopPolling,
    isPolling: !!intervalRef.current
  };
};

export default useApi;