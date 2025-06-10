// Formatters for Football Analytics Platform
// Centralized formatting functions used throughout the application

import {
    CONFEDERATION_LABELS,
    CURRENCY_SYMBOLS,
    DEFAULTS,
    INJURY_SEVERITY_LABELS,
    INJURY_TYPE_LABELS,
    MATCH_STATUSES,
    POSITION_LABELS,
    PREDICTION_CONFIDENCE_LABELS,
    PREDICTION_CONFIDENCE_LEVELS,
    PREDICTION_OUTCOME_LABELS,
    SUBSCRIPTION_TYPE_LABELS,
    USER_ROLE_LABELS
} from './constants';

// =============================================================================
// DATE AND TIME FORMATTERS
// =============================================================================

/**
 * Format date using various formats
 * @param {Date|string|number} date - Date to format
 * @param {string} format - Format type from DATE_FORMATS
 * @param {string} locale - Locale for formatting (default: 'en-US')
 * @param {string} timeZone - Timezone for formatting
 * @returns {string} Formatted date string
 */
export const formatDate = (date, format = 'MEDIUM', locale = 'en-US', timeZone = DEFAULTS.TIMEZONE) => {
  if (!date) return '';
  
  const dateObj = new Date(date);
  if (isNaN(dateObj.getTime())) return '';
  
  const options = { timeZone };
  
  switch (format) {
    case 'SHORT':
      return dateObj.toLocaleDateString(locale, { ...options, dateStyle: 'short' });
    case 'MEDIUM':
      return dateObj.toLocaleDateString(locale, { ...options, dateStyle: 'medium' });
    case 'LONG':
      return dateObj.toLocaleDateString(locale, { ...options, dateStyle: 'long' });
    case 'FULL':
      return dateObj.toLocaleDateString(locale, { ...options, dateStyle: 'full' });
    case 'ISO':
      return dateObj.toISOString().split('T')[0];
    case 'TIME':
      return dateObj.toLocaleTimeString(locale, { ...options, timeStyle: 'short' });
    case 'DATETIME':
      return dateObj.toLocaleString(locale, { ...options, dateStyle: 'short', timeStyle: 'short' });
    case 'TIMESTAMP':
      return dateObj.toLocaleString(locale, { ...options, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    default:
      return dateObj.toLocaleDateString(locale, options);
  }
};

/**
 * Format time duration in various formats
 * @param {number} milliseconds - Duration in milliseconds
 * @param {string} format - Format type ('short', 'medium', 'long')
 * @returns {string} Formatted duration string
 */
export const formatDuration = (milliseconds, format = 'medium') => {
  if (!milliseconds || milliseconds < 0) return '0m';
  
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (format === 'short') {
    if (days > 0) return `${days}d`;
    if (hours > 0) return `${hours}h`;
    if (minutes > 0) return `${minutes}m`;
    return `${seconds}s`;
  }
  
  if (format === 'long') {
    const parts = [];
    if (days > 0) parts.push(`${days} day${days === 1 ? '' : 's'}`);
    if (hours > 0) parts.push(`${hours % 24} hour${hours % 24 === 1 ? '' : 's'}`);
    if (minutes > 0) parts.push(`${minutes % 60} minute${minutes % 60 === 1 ? '' : 's'}`);
    if (parts.length === 0 && seconds > 0) parts.push(`${seconds} second${seconds === 1 ? '' : 's'}`);
    return parts.join(', ') || '0 seconds';
  }
  
  // Medium format (default)
  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours % 24}h`);
  if (minutes > 0) parts.push(`${minutes % 60}m`);
  if (parts.length === 0 && seconds > 0) parts.push(`${seconds}s`);
  return parts.join(' ') || '0m';
};

/**
 * Format relative time (e.g., "2 hours ago", "in 3 days")
 * @param {Date|string|number} date - Date to format
 * @param {string} locale - Locale for formatting
 * @returns {string} Relative time string
 */
export const formatRelativeTime = (date, locale = 'en-US') => {
  if (!date) return '';
  
  const dateObj = new Date(date);
  const now = new Date();
  const diffMs = dateObj.getTime() - now.getTime();
  
  // Use Intl.RelativeTimeFormat for modern browsers
  if (typeof Intl !== 'undefined' && Intl.RelativeTimeFormat) {
    const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
    
    const diffSeconds = Math.round(diffMs / 1000);
    const diffMinutes = Math.round(diffMs / (1000 * 60));
    const diffHours = Math.round(diffMs / (1000 * 60 * 60));
    const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));
    
    if (Math.abs(diffSeconds) < 60) return rtf.format(diffSeconds, 'second');
    if (Math.abs(diffMinutes) < 60) return rtf.format(diffMinutes, 'minute');
    if (Math.abs(diffHours) < 24) return rtf.format(diffHours, 'hour');
    if (Math.abs(diffDays) < 30) return rtf.format(diffDays, 'day');
    
    const diffMonths = Math.round(diffDays / 30);
    if (Math.abs(diffMonths) < 12) return rtf.format(diffMonths, 'month');
    
    const diffYears = Math.round(diffDays / 365);
    return rtf.format(diffYears, 'year');
  }
  
  // Fallback for older browsers
  const diffSeconds = Math.abs(Math.round(diffMs / 1000));
  const isFuture = diffMs > 0;
  
  if (diffSeconds < 60) return isFuture ? 'in a few seconds' : 'a few seconds ago';
  
  const diffMinutes = Math.round(diffSeconds / 60);
  if (diffMinutes < 60) {
    return isFuture ? `in ${diffMinutes} minute${diffMinutes === 1 ? '' : 's'}` : `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`;
  }
  
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) {
    return isFuture ? `in ${diffHours} hour${diffHours === 1 ? '' : 's'}` : `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
  }
  
  const diffDays = Math.round(diffHours / 24);
  return isFuture ? `in ${diffDays} day${diffDays === 1 ? '' : 's'}` : `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
};

/**
 * Format match time (e.g., "45+2'", "90'", "HT")
 * @param {number} minute - Match minute
 * @param {number} extraTime - Extra time minutes
 * @param {string} period - Match period ('first_half', 'second_half', 'extra_time', 'penalties')
 * @returns {string} Formatted match time
 */
export const formatMatchTime = (minute, extraTime = 0, period = 'first_half') => {
  if (!minute && minute !== 0) return '';
  
  switch (period) {
    case 'first_half':
      if (minute === 45 && extraTime > 0) return `45+${extraTime}'`;
      return `${minute}'`;
    case 'second_half':
      if (minute === 90 && extraTime > 0) return `90+${extraTime}'`;
      return `${minute}'`;
    case 'extra_time':
      if (minute <= 105) return `${minute}'`;
      if (minute === 105 && extraTime > 0) return `105+${extraTime}'`;
      if (minute <= 120) return `${minute}'`;
      if (minute === 120 && extraTime > 0) return `120+${extraTime}'`;
      return `${minute}'`;
    case 'penalties':
      return 'Pen.';
    case 'half_time':
      return 'HT';
    case 'full_time':
      return 'FT';
    default:
      return `${minute}'`;
  }
};

// =============================================================================
// NUMBER AND CURRENCY FORMATTERS
// =============================================================================

/**
 * Format currency values
 * @param {number} amount - Amount to format
 * @param {string} currency - Currency code
 * @param {string} locale - Locale for formatting
 * @param {object} options - Additional formatting options
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (amount, currency = DEFAULTS.CURRENCY, locale = 'en-US', options = {}) => {
  if (amount === null || amount === undefined || isNaN(amount)) return 'N/A';
  
  const defaultOptions = {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: amount >= 1000000 ? 1 : 2,
    ...options
  };
  
  try {
    return new Intl.NumberFormat(locale, defaultOptions).format(amount);
  } catch (error) {
    // Fallback formatting
    const symbol = CURRENCY_SYMBOLS[currency] || currency;
    return `${symbol}${formatNumber(amount, locale, { maximumFractionDigits: defaultOptions.maximumFractionDigits })}`;
  }
};

/**
 * Format market value with appropriate units (K, M, B)
 * @param {number} value - Market value to format
 * @param {string} currency - Currency code
 * @param {string} locale - Locale for formatting
 * @returns {string} Formatted market value
 */
export const formatMarketValue = (value, currency = DEFAULTS.CURRENCY, locale = 'en-US') => {
  if (value === null || value === undefined || isNaN(value) || value === 0) return 'N/A';
  
  const symbol = CURRENCY_SYMBOLS[currency] || currency;
  
  if (value >= 1000000000) {
    return `${symbol}${(value / 1000000000).toFixed(1)}B`;
  } else if (value >= 1000000) {
    return `${symbol}${(value / 1000000).toFixed(1)}M`;
  } else if (value >= 1000) {
    return `${symbol}${(value / 1000).toFixed(0)}K`;
  } else {
    return formatCurrency(value, currency, locale);
  }
};

/**
 * Format numbers with locale-specific formatting
 * @param {number} number - Number to format
 * @param {string} locale - Locale for formatting
 * @param {object} options - Formatting options
 * @returns {string} Formatted number string
 */
export const formatNumber = (number, locale = 'en-US', options = {}) => {
  if (number === null || number === undefined || isNaN(number)) return '0';
  
  try {
    return new Intl.NumberFormat(locale, options).format(number);
  } catch (error) {
    return number.toString();
  }
};

/**
 * Format percentage values
 * @param {number} value - Value to format as percentage (0-1 or 0-100)
 * @param {boolean} isDecimal - Whether input is decimal (0-1) or percentage (0-100)
 * @param {number} decimals - Number of decimal places
 * @param {string} locale - Locale for formatting
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value, isDecimal = true, decimals = 1, locale = 'en-US') => {
  if (value === null || value === undefined || isNaN(value)) return '0%';
  
  const percentage = isDecimal ? value * 100 : value;
  
  try {
    return new Intl.NumberFormat(locale, {
      style: 'percent',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(isDecimal ? value : value / 100);
  } catch (error) {
    return `${percentage.toFixed(decimals)}%`;
  }
};

/**
 * Format large numbers with abbreviations (K, M, B)
 * @param {number} number - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number with abbreviation
 */
export const formatNumberAbbreviation = (number, decimals = 1) => {
  if (number === null || number === undefined || isNaN(number)) return '0';
  
  if (number >= 1000000000) {
    return `${(number / 1000000000).toFixed(decimals)}B`;
  } else if (number >= 1000000) {
    return `${(number / 1000000).toFixed(decimals)}M`;
  } else if (number >= 1000) {
    return `${(number / 1000).toFixed(decimals)}K`;
  } else {
    return number.toString();
  }
};

// =============================================================================
// FOOTBALL-SPECIFIC FORMATTERS
// =============================================================================

/**
 * Format player name with proper capitalization
 * @param {string} firstName - Player's first name
 * @param {string} lastName - Player's last name
 * @param {string} format - Format type ('full', 'last_first', 'initials', 'short')
 * @returns {string} Formatted player name
 */
export const formatPlayerName = (firstName, lastName, format = 'full') => {
  if (!firstName && !lastName) return 'Unknown Player';
  
  const formatName = (name) => {
    if (!name) return '';
    return name.split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };
  
  const formattedFirst = formatName(firstName);
  const formattedLast = formatName(lastName);
  
  switch (format) {
    case 'full':
      return [formattedFirst, formattedLast].filter(Boolean).join(' ');
    case 'last_first':
      return [formattedLast, formattedFirst].filter(Boolean).join(', ');
    case 'initials':
      const firstInitial = formattedFirst ? formattedFirst.charAt(0) + '.' : '';
      return [firstInitial, formattedLast].filter(Boolean).join(' ');
    case 'short':
      return formattedLast || formattedFirst;
    default:
      return [formattedFirst, formattedLast].filter(Boolean).join(' ');
  }
};

/**
 * Format team name
 * @param {string} name - Team name
 * @param {string} abbreviation - Team abbreviation
 * @param {string} format - Format type ('full', 'short', 'abbreviation')
 * @returns {string} Formatted team name
 */
export const formatTeamName = (name, abbreviation, format = 'full') => {
  if (!name) return 'Unknown Team';
  
  switch (format) {
    case 'full':
      return name;
    case 'short':
      // Return first 3 words or abbreviation
      return abbreviation || name.split(' ').slice(0, 3).join(' ');
    case 'abbreviation':
      return abbreviation || name.split(' ').map(word => word.charAt(0)).join('').slice(0, 3).toUpperCase();
    default:
      return name;
  }
};

/**
 * Format match score
 * @param {number} homeScore - Home team score
 * @param {number} awayScore - Away team score
 * @param {string} status - Match status
 * @returns {string} Formatted score string
 */
export const formatMatchScore = (homeScore, awayScore, status) => {
  if (status === MATCH_STATUSES.UPCOMING) {
    return 'vs';
  }
  
  if (homeScore === null || homeScore === undefined || awayScore === null || awayScore === undefined) {
    return '-';
  }
  
  return `${homeScore} - ${awayScore}`;
};

/**
 * Format player position
 * @param {string} position - Position code
 * @param {string} format - Format type ('code', 'label', 'full')
 * @returns {string} Formatted position
 */
export const formatPosition = (position, format = 'label') => {
  if (!position) return 'Unknown';
  
  const upperPosition = position.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperPosition;
    case 'label':
      return POSITION_LABELS[upperPosition] || upperPosition;
    case 'full':
      return `${upperPosition} - ${POSITION_LABELS[upperPosition] || 'Unknown Position'}`;
    default:
      return POSITION_LABELS[upperPosition] || upperPosition;
  }
};

/**
 * Format confederation
 * @param {string} confederation - Confederation code
 * @param {string} format - Format type ('code', 'label', 'full')
 * @returns {string} Formatted confederation
 */
export const formatConfederation = (confederation, format = 'label') => {
  if (!confederation) return 'Unknown';
  
  const upperConfederation = confederation.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperConfederation;
    case 'label':
      return CONFEDERATION_LABELS[upperConfederation] || upperConfederation;
    case 'full':
      return `${upperConfederation} - ${CONFEDERATION_LABELS[upperConfederation] || 'Unknown Confederation'}`;
    default:
      return CONFEDERATION_LABELS[upperConfederation] || upperConfederation;
  }
};

/**
 * Calculate and format player age
 * @param {Date|string} dateOfBirth - Player's date of birth
 * @param {Date|string} asOfDate - Date to calculate age as of (default: today)
 * @returns {number|null} Player age in years
 */
export const calculateAge = (dateOfBirth, asOfDate = new Date()) => {
  if (!dateOfBirth) return null;
  
  const birthDate = new Date(dateOfBirth);
  const currentDate = new Date(asOfDate);
  
  if (isNaN(birthDate.getTime()) || isNaN(currentDate.getTime())) return null;
  
  let age = currentDate.getFullYear() - birthDate.getFullYear();
  const monthDiff = currentDate.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && currentDate.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age >= 0 ? age : null;
};

/**
 * Format stadium capacity
 * @param {number} capacity - Stadium capacity
 * @param {string} format - Format type ('number', 'abbreviated', 'full')
 * @returns {string} Formatted capacity
 */
export const formatStadiumCapacity = (capacity, format = 'number') => {
  if (!capacity || capacity <= 0) return 'N/A';
  
  switch (format) {
    case 'number':
      return formatNumber(capacity);
    case 'abbreviated':
      return formatNumberAbbreviation(capacity, 0);
    case 'full':
      return `${formatNumber(capacity)} capacity`;
    default:
      return formatNumber(capacity);
  }
};

// =============================================================================
// INJURY FORMATTERS
// =============================================================================

/**
 * Format injury type
 * @param {string} injuryType - Injury type code
 * @param {string} format - Format type ('code', 'label')
 * @returns {string} Formatted injury type
 */
export const formatInjuryType = (injuryType, format = 'label') => {
  if (!injuryType) return 'Unknown';
  
  const upperType = injuryType.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperType;
    case 'label':
      return INJURY_TYPE_LABELS[upperType] || upperType;
    default:
      return INJURY_TYPE_LABELS[upperType] || upperType;
  }
};

/**
 * Format injury severity
 * @param {string} severity - Injury severity code
 * @param {string} format - Format type ('code', 'label')
 * @returns {string} Formatted injury severity
 */
export const formatInjurySeverity = (severity, format = 'label') => {
  if (!severity) return 'Unknown';
  
  const upperSeverity = severity.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperSeverity;
    case 'label':
      return INJURY_SEVERITY_LABELS[upperSeverity] || upperSeverity;
    default:
      return INJURY_SEVERITY_LABELS[upperSeverity] || upperSeverity;
  }
};

/**
 * Format recovery time
 * @param {Date|string} expectedReturn - Expected return date
 * @param {Date|string} injuryDate - Injury date
 * @returns {string} Formatted recovery time
 */
export const formatRecoveryTime = (expectedReturn, injuryDate = new Date()) => {
  if (!expectedReturn) return 'Unknown';
  
  const returnDate = new Date(expectedReturn);
  const startDate = new Date(injuryDate);
  
  if (isNaN(returnDate.getTime())) return 'Unknown';
  
  const now = new Date();
  const totalRecoveryMs = returnDate.getTime() - startDate.getTime();
  const remainingMs = returnDate.getTime() - now.getTime();
  
  if (remainingMs <= 0) {
    return 'Ready to return';
  }
  
  const remainingDays = Math.ceil(remainingMs / (1000 * 60 * 60 * 24));
  const totalDays = Math.ceil(totalRecoveryMs / (1000 * 60 * 60 * 24));
  
  if (remainingDays === 1) {
    return '1 day remaining';
  } else if (remainingDays < 7) {
    return `${remainingDays} days remaining`;
  } else if (remainingDays < 30) {
    const weeks = Math.ceil(remainingDays / 7);
    return `${weeks} week${weeks === 1 ? '' : 's'} remaining`;
  } else {
    const months = Math.ceil(remainingDays / 30);
    return `${months} month${months === 1 ? '' : 's'} remaining`;
  }
};

// =============================================================================
// PREDICTION FORMATTERS
// =============================================================================

/**
 * Format prediction outcome
 * @param {string} outcome - Prediction outcome code
 * @param {string} format - Format type ('code', 'label')
 * @returns {string} Formatted prediction outcome
 */
export const formatPredictionOutcome = (outcome, format = 'label') => {
  if (!outcome) return 'Unknown';
  
  const upperOutcome = outcome.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperOutcome;
    case 'label':
      return PREDICTION_OUTCOME_LABELS[upperOutcome] || upperOutcome;
    default:
      return PREDICTION_OUTCOME_LABELS[upperOutcome] || upperOutcome;
  }
};

/**
 * Format prediction confidence
 * @param {number} confidence - Confidence percentage (0-100)
 * @param {string} format - Format type ('percentage', 'level', 'full')
 * @returns {string} Formatted confidence
 */
export const formatPredictionConfidence = (confidence, format = 'percentage') => {
  if (confidence === null || confidence === undefined || isNaN(confidence)) return 'Unknown';
  
  const level = confidence >= 80 ? PREDICTION_CONFIDENCE_LEVELS.HIGH :
                confidence >= 60 ? PREDICTION_CONFIDENCE_LEVELS.MEDIUM :
                PREDICTION_CONFIDENCE_LEVELS.LOW;
  
  switch (format) {
    case 'percentage':
      return `${Math.round(confidence)}%`;
    case 'level':
      return PREDICTION_CONFIDENCE_LABELS[level];
    case 'full':
      return `${Math.round(confidence)}% (${PREDICTION_CONFIDENCE_LABELS[level]})`;
    default:
      return `${Math.round(confidence)}%`;
  }
};

/**
 * Format prediction probabilities
 * @param {object} probabilities - Object with home, draw, away probabilities
 * @param {string} format - Format type ('percentages', 'decimals')
 * @returns {object} Formatted probabilities
 */
export const formatPredictionProbabilities = (probabilities, format = 'percentages') => {
  if (!probabilities) return { home: 'N/A', draw: 'N/A', away: 'N/A' };
  
  const { home = 0, draw = 0, away = 0 } = probabilities;
  
  if (format === 'decimals') {
    return {
      home: (home / 100).toFixed(3),
      draw: (draw / 100).toFixed(3),
      away: (away / 100).toFixed(3)
    };
  }
  
  return {
    home: `${Math.round(home)}%`,
    draw: `${Math.round(draw)}%`,
    away: `${Math.round(away)}%`
  };
};

// =============================================================================
// USER AND ROLE FORMATTERS
// =============================================================================

/**
 * Format user role
 * @param {string} role - User role code
 * @param {string} format - Format type ('code', 'label')
 * @returns {string} Formatted user role
 */
export const formatUserRole = (role, format = 'label') => {
  if (!role) return 'Unknown';
  
  const upperRole = role.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperRole;
    case 'label':
      return USER_ROLE_LABELS[upperRole] || upperRole;
    default:
      return USER_ROLE_LABELS[upperRole] || upperRole;
  }
};

/**
 * Format subscription type
 * @param {string} subscriptionType - Subscription type code
 * @param {string} format - Format type ('code', 'label')
 * @returns {string} Formatted subscription type
 */
export const formatSubscriptionType = (subscriptionType, format = 'label') => {
  if (!subscriptionType) return 'Unknown';
  
  const upperType = subscriptionType.toUpperCase();
  
  switch (format) {
    case 'code':
      return upperType;
    case 'label':
      return SUBSCRIPTION_TYPE_LABELS[upperType] || upperType;
    default:
      return SUBSCRIPTION_TYPE_LABELS[upperType] || upperType;
  }
};

// =============================================================================
// URL AND SLUG FORMATTERS
// =============================================================================

/**
 * Create URL-friendly slug from text
 * @param {string} text - Text to convert to slug
 * @param {number} maxLength - Maximum length of slug
 * @returns {string} URL-friendly slug
 */
export const createSlug = (text, maxLength = 50) => {
  if (!text) return '';
  
  return text
    .toString()
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '-')           // Replace spaces with -
    .replace(/[^\w\-]+/g, '')       // Remove all non-word chars
    .replace(/\-\-+/g, '-')         // Replace multiple - with single -
    .replace(/^-+/, '')             // Trim - from start of text
    .replace(/-+$/, '')             // Trim - from end of text
    .slice(0, maxLength);
};

/**
 * Format file size in human-readable format
 * @param {number} bytes - File size in bytes
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 B';
  if (!bytes || bytes < 0) return 'N/A';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// =============================================================================
// VALIDATION AND SANITIZATION
// =============================================================================

/**
 * Sanitize text input
 * @param {string} text - Text to sanitize
 * @param {object} options - Sanitization options
 * @returns {string} Sanitized text
 */
export const sanitizeText = (text, options = {}) => {
  if (!text || typeof text !== 'string') return '';
  
  const {
    allowHTML = false,
    maxLength = null,
    trimWhitespace = true,
    removeExtraSpaces = true
  } = options;
  
  let sanitized = text;
  
  // Remove HTML tags if not allowed
  if (!allowHTML) {
    sanitized = sanitized.replace(/<[^>]*>/g, '');
  }
  
  // Trim whitespace
  if (trimWhitespace) {
    sanitized = sanitized.trim();
  }
  
  // Remove extra spaces
  if (removeExtraSpaces) {
    sanitized = sanitized.replace(/\s+/g, ' ');
  }
  
  // Limit length
  if (maxLength && sanitized.length > maxLength) {
    sanitized = sanitized.slice(0, maxLength).trim();
  }
  
  return sanitized;
};

/**
 * Truncate text with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @param {string} suffix - Suffix to add (default: '...')
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 100, suffix = '...') => {
  if (!text || typeof text !== 'string') return '';
  if (text.length <= maxLength) return text;
  
  return text.slice(0, maxLength - suffix.length).trim() + suffix;
};

// =============================================================================
// STATISTICS FORMATTERS
// =============================================================================

/**
 * Format win/loss record
 * @param {number} wins - Number of wins
 * @param {number} draws - Number of draws
 * @param {number} losses - Number of losses
 * @param {string} format - Format type ('short', 'long', 'percentage')
 * @returns {string} Formatted record
 */
export const formatRecord = (wins = 0, draws = 0, losses = 0, format = 'short') => {
  const total = wins + draws + losses;
  
  if (total === 0) return format === 'percentage' ? '0%' : '0-0-0';
  
  switch (format) {
    case 'short':
      return `${wins}-${draws}-${losses}`;
    case 'long':
      return `${wins} W, ${draws} D, ${losses} L`;
    case 'percentage':
      const winPercentage = ((wins / total) * 100).toFixed(1);
      return `${winPercentage}%`;
    default:
      return `${wins}-${draws}-${losses}`;
  }
};

/**
 * Format goals for/against
 * @param {number} goalsFor - Goals scored
 * @param {number} goalsAgainst - Goals conceded
 * @param {string} format - Format type ('ratio', 'difference', 'separate')
 * @returns {string} Formatted goals
 */
export const formatGoalsForAgainst = (goalsFor = 0, goalsAgainst = 0, format = 'ratio') => {
  switch (format) {
    case 'ratio':
      return `${goalsFor}:${goalsAgainst}`;
    case 'difference':
      const diff = goalsFor - goalsAgainst;
      return diff >= 0 ? `+${diff}` : `${diff}`;
    case 'separate':
      return `${goalsFor} GF, ${goalsAgainst} GA`;
    default:
      return `${goalsFor}:${goalsAgainst}`;
  }
};

// =============================================================================
// EXPORT ALL FORMATTERS
// =============================================================================

export default {
  // Date and time
  formatDate,
  formatDuration,
  formatRelativeTime,
  formatMatchTime,
  
  // Numbers and currency
  formatCurrency,
  formatMarketValue,
  formatNumber,
  formatPercentage,
  formatNumberAbbreviation,
  
  // Football-specific
  formatPlayerName,
  formatTeamName,
  formatMatchScore,
  formatPosition,
  formatConfederation,
  calculateAge,
  formatStadiumCapacity,
  
  // Injuries
  formatInjuryType,
  formatInjurySeverity,
  formatRecoveryTime,
  
  // Predictions
  formatPredictionOutcome,
  formatPredictionConfidence,
  formatPredictionProbabilities,
  
  // Users and roles
  formatUserRole,
  formatSubscriptionType,
  
  // URLs and files
  createSlug,
  formatFileSize,
  
  // Validation and sanitization
  sanitizeText,
  truncateText,
  
  // Statistics
  formatRecord,
  formatGoalsForAgainst
};