// Funciones utilitarias para Football Analytics Platform
// Helpers centralizados utilizados en toda la aplicación

import {
    MATCH_STATUSES,
    POSITION_COLORS,
    REGEX_PATTERNS,
    VALIDATION_RULES
} from './constants';

// =============================================================================
// VALIDADORES
// =============================================================================

/**
 * Valida una dirección de email
 * @param {string} email - Email a validar
 * @returns {boolean} - True si el email es válido
 */
export const isValidEmail = (email) => {
  if (!email || typeof email !== 'string') return false;
  return REGEX_PATTERNS.EMAIL.test(email.trim());
};

/**
 * Valida una contraseña según las reglas establecidas
 * @param {string} password - Contraseña a validar
 * @returns {object} - Objeto con isValid y array de errores
 */
export const validatePassword = (password) => {
  const errors = [];
  const rules = VALIDATION_RULES.PASSWORD;
  
  if (!password || typeof password !== 'string') {
    return { isValid: false, errors: ['La contraseña es requerida'] };
  }
  
  if (password.length < rules.MIN_LENGTH) {
    errors.push(`La contraseña debe tener al menos ${rules.MIN_LENGTH} caracteres`);
  }
  
  if (password.length > rules.MAX_LENGTH) {
    errors.push(`La contraseña no puede exceder ${rules.MAX_LENGTH} caracteres`);
  }
  
  if (rules.REQUIRE_UPPERCASE && !/[A-Z]/.test(password)) {
    errors.push('La contraseña debe contener al menos una letra mayúscula');
  }
  
  if (rules.REQUIRE_LOWERCASE && !/[a-z]/.test(password)) {
    errors.push('La contraseña debe contener al menos una letra minúscula');
  }
  
  if (rules.REQUIRE_NUMBERS && !/\d/.test(password)) {
    errors.push('La contraseña debe contener al menos un número');
  }
  
  if (rules.REQUIRE_SPECIAL_CHARS && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('La contraseña debe contener al menos un carácter especial');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Valida un número de teléfono
 * @param {string} phone - Número de teléfono a validar
 * @returns {boolean} - True si el teléfono es válido
 */
export const isValidPhone = (phone) => {
  if (!phone || typeof phone !== 'string') return false;
  const cleanPhone = phone.replace(/\s/g, '');
  return REGEX_PATTERNS.PHONE.test(cleanPhone) && 
         cleanPhone.length >= VALIDATION_RULES.PHONE.MIN_LENGTH &&
         cleanPhone.length <= VALIDATION_RULES.PHONE.MAX_LENGTH;
};

/**
 * Valida una URL
 * @param {string} url - URL a validar
 * @returns {boolean} - True si la URL es válida
 */
export const isValidURL = (url) => {
  if (!url || typeof url !== 'string') return false;
  return REGEX_PATTERNS.URL.test(url.trim());
};

/**
 * Valida si un valor es un UUID válido
 * @param {string} uuid - UUID a validar
 * @returns {boolean} - True si es un UUID válido
 */
export const isValidUUID = (uuid) => {
  if (!uuid || typeof uuid !== 'string') return false;
  return REGEX_PATTERNS.UUID.test(uuid);
};

// =============================================================================
// UTILIDADES DE ARRAYS Y OBJETOS
// =============================================================================

/**
 * Elimina elementos duplicados de un array
 * @param {Array} array - Array con posibles duplicados
 * @param {string} key - Clave para comparar objetos (opcional)
 * @returns {Array} - Array sin duplicados
 */
export const removeDuplicates = (array, key = null) => {
  if (!Array.isArray(array)) return [];
  
  if (key) {
    // Para arrays de objetos
    const seen = new Set();
    return array.filter(item => {
      const val = item[key];
      if (seen.has(val)) return false;
      seen.add(val);
      return true;
    });
  }
  
  // Para arrays primitivos
  return [...new Set(array)];
};

/**
 * Agrupa elementos de un array por una clave específica
 * @param {Array} array - Array a agrupar
 * @param {string|function} key - Clave o función para agrupar
 * @returns {object} - Objeto agrupado
 */
export const groupBy = (array, key) => {
  if (!Array.isArray(array)) return {};
  
  return array.reduce((groups, item) => {
    const groupKey = typeof key === 'function' ? key(item) : item[key];
    if (!groups[groupKey]) {
      groups[groupKey] = [];
    }
    groups[groupKey].push(item);
    return groups;
  }, {});
};

/**
 * Ordena un array de objetos por una clave específica
 * @param {Array} array - Array a ordenar
 * @param {string} key - Clave para ordenar
 * @param {string} order - Orden: 'asc' o 'desc'
 * @returns {Array} - Array ordenado
 */
export const sortBy = (array, key, order = 'asc') => {
  if (!Array.isArray(array)) return [];
  
  return [...array].sort((a, b) => {
    const aVal = getNestedValue(a, key);
    const bVal = getNestedValue(b, key);
    
    // Manejo de valores nulos/undefined
    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;
    
    // Comparación
    if (aVal < bVal) return order === 'asc' ? -1 : 1;
    if (aVal > bVal) return order === 'asc' ? 1 : -1;
    return 0;
  });
};

/**
 * Obtiene un valor anidado de un objeto usando notación de punto
 * @param {object} obj - Objeto del cual obtener el valor
 * @param {string} path - Ruta al valor (ej: 'user.profile.name')
 * @param {any} defaultValue - Valor por defecto si no existe
 * @returns {any} - Valor encontrado o valor por defecto
 */
export const getNestedValue = (obj, path, defaultValue = null) => {
  if (!obj || typeof obj !== 'object' || !path) return defaultValue;
  
  return path.split('.').reduce((current, key) => {
    return current && current[key] !== undefined ? current[key] : defaultValue;
  }, obj);
};

/**
 * Establece un valor anidado en un objeto usando notación de punto
 * @param {object} obj - Objeto a modificar
 * @param {string} path - Ruta donde establecer el valor
 * @param {any} value - Valor a establecer
 * @returns {object} - Objeto modificado
 */
export const setNestedValue = (obj, path, value) => {
  const keys = path.split('.');
  const result = { ...obj };
  let current = result;
  
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!current[key] || typeof current[key] !== 'object') {
      current[key] = {};
    }
    current = current[key];
  }
  
  current[keys[keys.length - 1]] = value;
  return result;
};

/**
 * Crea una copia profunda de un objeto o array
 * @param {any} obj - Objeto a clonar
 * @returns {any} - Copia profunda del objeto
 */
export const deepClone = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj;
  
  if (obj instanceof Date) return new Date(obj.getTime());
  if (obj instanceof Array) return obj.map(item => deepClone(item));
  
  if (typeof obj === 'object') {
    const cloned = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }
  
  return obj;
};

/**
 * Combina múltiples objetos de forma profunda
 * @param {...object} objects - Objetos a combinar
 * @returns {object} - Objeto combinado
 */
export const deepMerge = (...objects) => {
  const isObject = (obj) => obj && typeof obj === 'object' && !Array.isArray(obj);
  
  return objects.reduce((merged, obj) => {
    if (!isObject(obj)) return merged;
    
    Object.keys(obj).forEach(key => {
      if (isObject(obj[key]) && isObject(merged[key])) {
        merged[key] = deepMerge(merged[key], obj[key]);
      } else {
        merged[key] = obj[key];
      }
    });
    
    return merged;
  }, {});
};

// =============================================================================
// UTILIDADES DE RENDIMIENTO
// =============================================================================

/**
 * Función debounce para limitar la frecuencia de llamadas
 * @param {function} func - Función a ejecutar
 * @param {number} delay - Retraso en milisegundos
 * @returns {function} - Función debounced
 */
export const debounce = (func, delay) => {
  let timeoutId;
  
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), delay);
  };
};

/**
 * Función throttle para limitar la frecuencia de ejecución
 * @param {function} func - Función a ejecutar
 * @param {number} limit - Límite de tiempo en milisegundos
 * @returns {function} - Función throttled
 */
export const throttle = (func, limit) => {
  let inThrottle;
  
  return function (...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

/**
 * Memoriza el resultado de una función para optimizar rendimiento
 * @param {function} func - Función a memorizar
 * @param {function} keyGenerator - Función para generar la clave de cache
 * @returns {function} - Función memorizada
 */
export const memoize = (func, keyGenerator = (...args) => JSON.stringify(args)) => {
  const cache = new Map();
  
  return function (...args) {
    const key = keyGenerator(...args);
    
    if (cache.has(key)) {
      return cache.get(key);
    }
    
    const result = func.apply(this, args);
    cache.set(key, result);
    return result;
  };
};

// =============================================================================
// UTILIDADES DE FECHAS ESPECÍFICAS DE FÚTBOL
// =============================================================================

/**
 * Verifica si un partido está en vivo basado en fecha y estado
 * @param {string|Date} matchDate - Fecha del partido
 * @param {string} status - Estado del partido
 * @returns {boolean} - True si el partido está en vivo
 */
export const isMatchLive = (matchDate, status) => {
  if (status === MATCH_STATUSES.LIVE || status === MATCH_STATUSES.HALF_TIME) {
    return true;
  }
  
  if (status === MATCH_STATUSES.FINISHED) {
    return false;
  }
  
  // Verificar si el partido debería estar en vivo basado en la fecha
  const now = new Date();
  const match = new Date(matchDate);
  const diffMs = now.getTime() - match.getTime();
  const diffMinutes = diffMs / (1000 * 60);
  
  // Considerar un partido como posiblemente en vivo si está entre 0 y 120 minutos después del inicio
  return diffMinutes >= 0 && diffMinutes <= 120;
};

/**
 * Calcula los días hasta el próximo partido
 * @param {string|Date} matchDate - Fecha del partido
 * @returns {number} - Días hasta el partido (negativo si ya pasó)
 */
export const daysUntilMatch = (matchDate) => {
  const now = new Date();
  const match = new Date(matchDate);
  const diffTime = match.getTime() - now.getTime();
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

/**
 * Obtiene la temporada basada en una fecha
 * @param {string|Date} date - Fecha para determinar la temporada
 * @returns {string} - Temporada en formato "2024/25"
 */
export const getSeasonFromDate = (date) => {
  const dateObj = new Date(date);
  const year = dateObj.getFullYear();
  const month = dateObj.getMonth(); // 0-11
  
  // La temporada de fútbol típicamente va de agosto a mayo
  if (month >= 7) { // Agosto en adelante
    return `${year}/${(year + 1).toString().slice(-2)}`;
  } else { // Enero a julio
    return `${year - 1}/${year.toString().slice(-2)}`;
  }
};

/**
 * Verifica si una fecha está en la temporada actual
 * @param {string|Date} date - Fecha a verificar
 * @returns {boolean} - True si está en la temporada actual
 */
export const isCurrentSeason = (date) => {
  const currentSeason = getSeasonFromDate(new Date());
  const dateSeason = getSeasonFromDate(date);
  return currentSeason === dateSeason;
};

// =============================================================================
// UTILIDADES ESPECÍFICAS DE FÚTBOL
// =============================================================================

/**
 * Calcula el color de la posición de un jugador
 * @param {string} position - Posición del jugador
 * @returns {string} - Color hex para la posición
 */
export const getPositionColor = (position) => {
  if (!position) return '#6b7280'; // Gris por defecto
  
  const upperPosition = position.toUpperCase();
  
  // Mapeo de posiciones detalladas a principales
  if (['GK'].includes(upperPosition)) {
    return POSITION_COLORS.GK;
  } else if (['CB', 'LB', 'RB', 'LWB', 'RWB', 'SW', 'DEF'].includes(upperPosition)) {
    return POSITION_COLORS.DEF;
  } else if (['CDM', 'CM', 'CAM', 'LM', 'RM', 'LW', 'RW', 'MID'].includes(upperPosition)) {
    return POSITION_COLORS.MID;
  } else if (['CF', 'ST', 'LF', 'RF', 'SS', 'FWD'].includes(upperPosition)) {
    return POSITION_COLORS.FWD;
  }
  
  return POSITION_COLORS[upperPosition] || '#6b7280';
};

/**
 * Determina el resultado de un partido para un equipo específico
 * @param {number} teamScore - Goles del equipo
 * @param {number} opponentScore - Goles del oponente
 * @returns {string} - 'W' (victoria), 'D' (empate), 'L' (derrota)
 */
export const getMatchResult = (teamScore, opponentScore) => {
  if (teamScore === null || teamScore === undefined || 
      opponentScore === null || opponentScore === undefined) {
    return null;
  }
  
  if (teamScore > opponentScore) return 'W';
  if (teamScore < opponentScore) return 'L';
  return 'D';
};

/**
 * Calcula la forma reciente de un equipo basada en resultados
 * @param {Array} matches - Array de partidos con resultados
 * @param {string} teamId - ID del equipo
 * @param {number} count - Número de partidos a considerar
 * @returns {object} - Objeto con estadísticas de forma
 */
export const calculateTeamForm = (matches, teamId, count = 5) => {
  if (!Array.isArray(matches) || matches.length === 0) {
    return { form: '', wins: 0, draws: 0, losses: 0, points: 0 };
  }
  
  const recentMatches = matches
    .filter(match => match.status === MATCH_STATUSES.FINISHED)
    .slice(0, count);
  
  let wins = 0, draws = 0, losses = 0;
  let form = '';
  
  recentMatches.forEach(match => {
    const isHome = match.homeTeamId === teamId;
    const teamScore = isHome ? match.homeScore : match.awayScore;
    const opponentScore = isHome ? match.awayScore : match.homeScore;
    
    const result = getMatchResult(teamScore, opponentScore);
    
    switch (result) {
      case 'W':
        wins++;
        form += 'W';
        break;
      case 'D':
        draws++;
        form += 'D';
        break;
      case 'L':
        losses++;
        form += 'L';
        break;
    }
  });
  
  const points = (wins * 3) + draws;
  
  return {
    form,
    wins,
    draws,
    losses,
    points,
    percentage: recentMatches.length > 0 ? (points / (recentMatches.length * 3)) * 100 : 0
  };
};

/**
 * Calcula estadísticas básicas de un jugador
 * @param {Array} matches - Partidos del jugador
 * @returns {object} - Estadísticas calculadas
 */
export const calculatePlayerStats = (matches) => {
  if (!Array.isArray(matches) || matches.length === 0) {
    return {
      appearances: 0,
      goals: 0,
      assists: 0,
      yellowCards: 0,
      redCards: 0,
      minutesPlayed: 0,
      averageRating: 0
    };
  }
  
  const stats = matches.reduce((acc, match) => {
    if (match.minutesPlayed > 0) {
      acc.appearances++;
      acc.minutesPlayed += match.minutesPlayed;
    }
    
    acc.goals += match.goals || 0;
    acc.assists += match.assists || 0;
    acc.yellowCards += match.yellowCards || 0;
    acc.redCards += match.redCards || 0;
    
    if (match.rating) {
      acc.totalRating += match.rating;
      acc.ratedMatches++;
    }
    
    return acc;
  }, {
    appearances: 0,
    goals: 0,
    assists: 0,
    yellowCards: 0,
    redCards: 0,
    minutesPlayed: 0,
    totalRating: 0,
    ratedMatches: 0
  });
  
  stats.averageRating = stats.ratedMatches > 0 ? stats.totalRating / stats.ratedMatches : 0;
  delete stats.totalRating;
  delete stats.ratedMatches;
  
  return stats;
};

// =============================================================================
// UTILIDADES DE ANÁLISIS DE DATOS
// =============================================================================

/**
 * Calcula percentiles de un array de valores
 * @param {Array} values - Array de valores numéricos
 * @param {number} percentile - Percentil a calcular (0-100)
 * @returns {number} - Valor del percentil
 */
export const calculatePercentile = (values, percentile) => {
  if (!Array.isArray(values) || values.length === 0) return 0;
  
  const sorted = values.filter(v => typeof v === 'number' && !isNaN(v)).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  
  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  
  if (lower === upper) return sorted[lower];
  
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

/**
 * Calcula estadísticas descriptivas básicas
 * @param {Array} values - Array de valores numéricos
 * @returns {object} - Estadísticas descriptivas
 */
export const calculateDescriptiveStats = (values) => {
  if (!Array.isArray(values) || values.length === 0) {
    return { count: 0, mean: 0, median: 0, min: 0, max: 0, std: 0 };
  }
  
  const numbers = values.filter(v => typeof v === 'number' && !isNaN(v));
  
  if (numbers.length === 0) {
    return { count: 0, mean: 0, median: 0, min: 0, max: 0, std: 0 };
  }
  
  const sorted = [...numbers].sort((a, b) => a - b);
  const count = numbers.length;
  const sum = numbers.reduce((a, b) => a + b, 0);
  const mean = sum / count;
  
  const median = count % 2 === 0
    ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2
    : sorted[Math.floor(count / 2)];
  
  const variance = numbers.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
  const std = Math.sqrt(variance);
  
  return {
    count,
    mean,
    median,
    min: sorted[0],
    max: sorted[count - 1],
    std
  };
};

/**
 * Normaliza valores a una escala de 0-100
 * @param {Array} values - Array de valores a normalizar
 * @returns {Array} - Array de valores normalizados
 */
export const normalizeValues = (values) => {
  if (!Array.isArray(values) || values.length === 0) return [];
  
  const numbers = values.filter(v => typeof v === 'number' && !isNaN(v));
  if (numbers.length === 0) return values;
  
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const range = max - min;
  
  if (range === 0) return numbers.map(() => 50); // Todos los valores iguales
  
  return numbers.map(value => ((value - min) / range) * 100);
};

// =============================================================================
// UTILIDADES DE URL Y NAVEGACIÓN
// =============================================================================

/**
 * Construye una URL con parámetros de consulta
 * @param {string} baseUrl - URL base
 * @param {object} params - Parámetros a agregar
 * @returns {string} - URL completa con parámetros
 */
export const buildUrlWithParams = (baseUrl, params = {}) => {
  if (!params || Object.keys(params).length === 0) return baseUrl;
  
  const url = new URL(baseUrl, window.location.origin);
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      url.searchParams.append(key, value);
    }
  });
  
  return url.toString();
};

/**
 * Extrae parámetros de consulta de una URL
 * @param {string} url - URL de la cual extraer parámetros (opcional, usa window.location)
 * @returns {object} - Objeto con los parámetros
 */
export const getUrlParams = (url = window.location.href) => {
  const urlObj = new URL(url);
  const params = {};
  
  urlObj.searchParams.forEach((value, key) => {
    params[key] = value;
  });
  
  return params;
};

/**
 * Navega a una ruta con parámetros opcionales
 * @param {function} navigate - Función de navegación de React Router
 * @param {string} path - Ruta de destino
 * @param {object} params - Parámetros de consulta
 * @param {object} state - Estado a pasar
 */
export const navigateWithParams = (navigate, path, params = {}, state = null) => {
  const url = buildUrlWithParams(path, params);
  const options = state ? { state } : {};
  navigate(url, options);
};

// =============================================================================
// UTILIDADES DE COLORES Y CSS
// =============================================================================

/**
 * Convierte un color hex a RGB
 * @param {string} hex - Color en formato hex
 * @returns {object} - Objeto con valores r, g, b
 */
export const hexToRgb = (hex) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
};

/**
 * Convierte RGB a hex
 * @param {number} r - Valor rojo (0-255)
 * @param {number} g - Valor verde (0-255)
 * @param {number} b - Valor azul (0-255)
 * @returns {string} - Color en formato hex
 */
export const rgbToHex = (r, g, b) => {
  return '#' + [r, g, b].map(x => {
    const hex = Math.round(x).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
};

/**
 * Calcula el contraste entre dos colores
 * @param {string} color1 - Primer color en hex
 * @param {string} color2 - Segundo color en hex
 * @returns {number} - Ratio de contraste
 */
export const getContrastRatio = (color1, color2) => {
  const getLuminance = (hex) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return 0;
    
    const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  
  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  
  return (brightest + 0.05) / (darkest + 0.05);
};

/**
 * Determina si usar texto claro u oscuro basado en el fondo
 * @param {string} backgroundColor - Color de fondo en hex
 * @returns {string} - 'light' o 'dark'
 */
export const getTextColorForBackground = (backgroundColor) => {
  const whiteContrast = getContrastRatio(backgroundColor, '#ffffff');
  const blackContrast = getContrastRatio(backgroundColor, '#000000');
  
  return whiteContrast > blackContrast ? 'light' : 'dark';
};

// =============================================================================
// UTILIDADES DE LOCALSTORAGE
// =============================================================================

/**
 * Guarda datos en localStorage de forma segura
 * @param {string} key - Clave para guardar
 * @param {any} value - Valor a guardar
 * @returns {boolean} - True si se guardó exitosamente
 */
export const safeSetLocalStorage = (key, value) => {
  try {
    const serializedValue = JSON.stringify(value);
    localStorage.setItem(key, serializedValue);
    return true;
  } catch (error) {
    console.error('Error guardando en localStorage:', error);
    return false;
  }
};

/**
 * Obtiene datos de localStorage de forma segura
 * @param {string} key - Clave a obtener
 * @param {any} defaultValue - Valor por defecto si no existe
 * @returns {any} - Valor obtenido o valor por defecto
 */
export const safeGetLocalStorage = (key, defaultValue = null) => {
  try {
    const item = localStorage.getItem(key);
    if (item === null) return defaultValue;
    return JSON.parse(item);
  } catch (error) {
    console.error('Error obteniendo de localStorage:', error);
    return defaultValue;
  }
};

/**
 * Elimina una clave de localStorage de forma segura
 * @param {string} key - Clave a eliminar
 * @returns {boolean} - True si se eliminó exitosamente
 */
export const safeRemoveLocalStorage = (key) => {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error eliminando de localStorage:', error);
    return false;
  }
};

// =============================================================================
// UTILIDADES DE DETECCIÓN DE DISPOSITIVO
// =============================================================================

/**
 * Detecta si el dispositivo es móvil
 * @returns {boolean} - True si es móvil
 */
export const isMobile = () => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

/**
 * Detecta si el dispositivo es tablet
 * @returns {boolean} - True si es tablet
 */
export const isTablet = () => {
  return /iPad|Android.*tablet|Kindle|Silk/i.test(navigator.userAgent);
};

/**
 * Detecta si es dispositivo iOS
 * @returns {boolean} - True si es iOS
 */
export const isIOS = () => {
  return /iPad|iPhone|iPod/.test(navigator.userAgent);
};

/**
 * Detecta si es dispositivo Android
 * @returns {boolean} - True si es Android
 */
export const isAndroid = () => {
  return /Android/.test(navigator.userAgent);
};

/**
 * Obtiene información del dispositivo
 * @returns {object} - Información del dispositivo
 */
export const getDeviceInfo = () => {
  return {
    isMobile: isMobile(),
    isTablet: isTablet(),
    isDesktop: !isMobile() && !isTablet(),
    isIOS: isIOS(),
    isAndroid: isAndroid(),
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    cookieEnabled: navigator.cookieEnabled,
    onLine: navigator.onLine
  };
};

// =============================================================================
// UTILIDADES MATEMÁTICAS
// =============================================================================

/**
 * Genera un número aleatorio entre min y max
 * @param {number} min - Valor mínimo
 * @param {number} max - Valor máximo
 * @returns {number} - Número aleatorio
 */
export const randomBetween = (min, max) => {
  return Math.random() * (max - min) + min;
};

/**
 * Redondea un número a un número específico de decimales
 * @param {number} num - Número a redondear
 * @param {number} decimals - Número de decimales
 * @returns {number} - Número redondeado
 */
export const roundToDecimals = (num, decimals) => {
  return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
};

/**
 * Clamp un valor entre min y max
 * @param {number} value - Valor a clampear
 * @param {number} min - Valor mínimo
 * @param {number} max - Valor máximo
 * @returns {number} - Valor clampeado
 */
export const clamp = (value, min, max) => {
  return Math.min(Math.max(value, min), max);
};

/**
 * Convierte grados a radianes
 * @param {number} degrees - Grados
 * @returns {number} - Radianes
 */
export const degreesToRadians = (degrees) => {
  return degrees * (Math.PI / 180);
};

/**
 * Convierte radianes a grados
 * @param {number} radians - Radianes
 * @returns {number} - Grados
 */
export const radiansToDegrees = (radians) => {
  return radians * (180 / Math.PI);
};

// =============================================================================
// UTILIDADES DE ERROR Y LOGGING
// =============================================================================

/**
 * Logger personalizado con diferentes niveles
 * @param {string} level - Nivel del log ('info', 'warn', 'error', 'debug')
 * @param {string} message - Mensaje a loggear
 * @param {any} data - Datos adicionales
 */
export const logger = (level, message, data = null) => {
  if (process.env.NODE_ENV === 'production' && level === 'debug') {
    return; // No loggear debug en producción
  }
  
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${level.toUpperCase()}: ${message}`;
  
  switch (level) {
    case 'error':
      console.error(logMessage, data);
      break;
    case 'warn':
      console.warn(logMessage, data);
      break;
    case 'info':
      console.info(logMessage, data);
      break;
    case 'debug':
    default:
      console.log(logMessage, data);
      break;
  }
};

/**
 * Maneja errores de forma consistente
 * @param {Error} error - Error a manejar
 * @param {string} context - Contexto donde ocurrió el error
 * @param {function} fallback - Función de fallback opcional
 */
export const handleError = (error, context = 'Unknown', fallback = null) => {
  logger('error', `Error en ${context}`, {
    message: error.message,
    stack: error.stack,
    timestamp: new Date().toISOString()
  });
  
  if (fallback && typeof fallback === 'function') {
    try {
      fallback(error);
    } catch (fallbackError) {
      logger('error', 'Error en función de fallback', fallbackError);
    }
  }
};

// =============================================================================
// EXPORT DE TODAS LAS UTILIDADES
// =============================================================================

export default {
  // Validadores
  isValidEmail,
  validatePassword,
  isValidPhone,
  isValidURL,
  isValidUUID,
  
  // Arrays y objetos
  removeDuplicates,
  groupBy,
  sortBy,
  getNestedValue,
  setNestedValue,
  deepClone,
  deepMerge,
  
  // Rendimiento
  debounce,
  throttle,
  memoize,
  
  // Fechas específicas de fútbol
  isMatchLive,
  daysUntilMatch,
  getSeasonFromDate,
  isCurrentSeason,
  
  // Específicas de fútbol
  getPositionColor,
  getMatchResult,
  calculateTeamForm,
  calculatePlayerStats,
  
  // Análisis de datos
  calculatePercentile,
  calculateDescriptiveStats,
  normalizeValues,
  
  // URL y navegación
  buildUrlWithParams,
  getUrlParams,
  navigateWithParams,
  
  // Colores y CSS
  hexToRgb,
  rgbToHex,
  getContrastRatio,
  getTextColorForBackground,
  
  // localStorage
  safeSetLocalStorage,
  safeGetLocalStorage,
  safeRemoveLocalStorage,
  
  // Detección de dispositivo
  isMobile,
  isTablet,
  isIOS,
  isAndroid,
  getDeviceInfo,
  
  // Matemáticas
  randomBetween,
  roundToDecimals,
  clamp,
  degreesToRadians,
  radiansToDegrees,
  
  // Error y logging
  logger,
  handleError
};