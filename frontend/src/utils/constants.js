// Constants for Football Analytics Platform
// Centralized constant definitions used throughout the application

// =============================================================================
// APPLICATION CONFIGURATION
// =============================================================================

export const APP_CONFIG = {
  NAME: 'Football Analytics',
  VERSION: '1.0.0',
  DESCRIPTION: 'Advanced football prediction and analytics platform with machine learning',
  AUTHOR: 'Football Analytics Team',
  COPYRIGHT: '© 2025 Football Analytics. All rights reserved.',
  
  // Environment
  ENVIRONMENT: process.env.NODE_ENV || 'development',
  
  // URLs
  BASE_URL: process.env.REACT_APP_BASE_URL || 'http://localhost:3000',
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:3001',
  
  // Social & External
  WEBSITE_URL: 'https://football-analytics.com',
  SUPPORT_EMAIL: 'support@football-analytics.com',
  DOCUMENTATION_URL: 'https://docs.football-analytics.com',
  
  // Analytics
  GOOGLE_ANALYTICS_ID: process.env.REACT_APP_GA_ID,
  HOTJAR_ID: process.env.REACT_APP_HOTJAR_ID,
  
  // Feature flags
  ENABLE_ANALYTICS: process.env.REACT_APP_ENABLE_ANALYTICS === 'true',
  ENABLE_WEBSOCKETS: process.env.REACT_APP_ENABLE_WEBSOCKETS !== 'false',
  ENABLE_NOTIFICATIONS: process.env.REACT_APP_ENABLE_NOTIFICATIONS !== 'false',
  ENABLE_PWA: process.env.REACT_APP_ENABLE_PWA === 'true'
};

// =============================================================================
// FOOTBALL DOMAIN CONSTANTS
// =============================================================================

// Confederations
export const CONFEDERATIONS = {
  UEFA: 'UEFA',
  CONMEBOL: 'CONMEBOL',
  CONCACAF: 'CONCACAF',
  AFC: 'AFC',
  CAF: 'CAF',
  OFC: 'OFC'
};

export const CONFEDERATION_LABELS = {
  [CONFEDERATIONS.UEFA]: 'Union of European Football Associations',
  [CONFEDERATIONS.CONMEBOL]: 'South American Football Confederation',
  [CONFEDERATIONS.CONCACAF]: 'Confederation of North, Central America and Caribbean Association Football',
  [CONFEDERATIONS.AFC]: 'Asian Football Confederation',
  [CONFEDERATIONS.CAF]: 'Confederation of African Football',
  [CONFEDERATIONS.OFC]: 'Oceania Football Confederation'
};

export const CONFEDERATION_COLORS = {
  [CONFEDERATIONS.UEFA]: '#003f7f',
  [CONFEDERATIONS.CONMEBOL]: '#ffcc00',
  [CONFEDERATIONS.CONCACAF]: '#c41e3a',
  [CONFEDERATIONS.AFC]: '#ff6600',
  [CONFEDERATIONS.CAF]: '#009639',
  [CONFEDERATIONS.OFC]: '#0066cc'
};

// Player positions
export const POSITIONS = {
  GK: 'GK',
  DEF: 'DEF',
  MID: 'MID',
  FWD: 'FWD'
};

export const POSITION_LABELS = {
  [POSITIONS.GK]: 'Goalkeeper',
  [POSITIONS.DEF]: 'Defender',
  [POSITIONS.MID]: 'Midfielder',
  [POSITIONS.FWD]: 'Forward'
};

export const POSITION_COLORS = {
  [POSITIONS.GK]: '#fbbf24',     // Yellow
  [POSITIONS.DEF]: '#3b82f6',    // Blue
  [POSITIONS.MID]: '#10b981',    // Green
  [POSITIONS.FWD]: '#ef4444'     // Red
};

export const DETAILED_POSITIONS = {
  // Goalkeepers
  GK: 'Goalkeeper',
  
  // Defenders
  CB: 'Centre-Back',
  LB: 'Left-Back',
  RB: 'Right-Back',
  LWB: 'Left Wing-Back',
  RWB: 'Right Wing-Back',
  SW: 'Sweeper',
  
  // Midfielders
  CDM: 'Defensive Midfielder',
  CM: 'Central Midfielder',
  CAM: 'Attacking Midfielder',
  LM: 'Left Midfielder',
  RM: 'Right Midfielder',
  LW: 'Left Winger',
  RW: 'Right Winger',
  
  // Forwards
  CF: 'Centre-Forward',
  ST: 'Striker',
  LF: 'Left Forward',
  RF: 'Right Forward',
  SS: 'Second Striker'
};

// Match statuses
export const MATCH_STATUSES = {
  UPCOMING: 'upcoming',
  LIVE: 'live',
  HALF_TIME: 'half_time',
  FINISHED: 'finished',
  POSTPONED: 'postponed',
  CANCELLED: 'cancelled',
  SUSPENDED: 'suspended'
};

export const MATCH_STATUS_LABELS = {
  [MATCH_STATUSES.UPCOMING]: 'Upcoming',
  [MATCH_STATUSES.LIVE]: 'Live',
  [MATCH_STATUSES.HALF_TIME]: 'Half Time',
  [MATCH_STATUSES.FINISHED]: 'Finished',
  [MATCH_STATUSES.POSTPONED]: 'Postponed',
  [MATCH_STATUSES.CANCELLED]: 'Cancelled',
  [MATCH_STATUSES.SUSPENDED]: 'Suspended'
};

export const MATCH_STATUS_COLORS = {
  [MATCH_STATUSES.UPCOMING]: '#6b7280',   // Gray
  [MATCH_STATUSES.LIVE]: '#ef4444',       // Red
  [MATCH_STATUSES.HALF_TIME]: '#f59e0b',  // Yellow
  [MATCH_STATUSES.FINISHED]: '#10b981',   // Green
  [MATCH_STATUSES.POSTPONED]: '#8b5cf6',  // Purple
  [MATCH_STATUSES.CANCELLED]: '#f87171',  // Light red
  [MATCH_STATUSES.SUSPENDED]: '#fb923c'   // Orange
};

// Match events
export const MATCH_EVENTS = {
  GOAL: 'goal',
  YELLOW_CARD: 'yellow_card',
  RED_CARD: 'red_card',
  SUBSTITUTION: 'substitution',
  PENALTY: 'penalty',
  OWN_GOAL: 'own_goal',
  VAR_DECISION: 'var_decision',
  INJURY: 'injury'
};

export const MATCH_EVENT_ICONS = {
  [MATCH_EVENTS.GOAL]: '⚽',
  [MATCH_EVENTS.YELLOW_CARD]: '🟨',
  [MATCH_EVENTS.RED_CARD]: '🟥',
  [MATCH_EVENTS.SUBSTITUTION]: '🔄',
  [MATCH_EVENTS.PENALTY]: '🥅',
  [MATCH_EVENTS.OWN_GOAL]: '⚽️',
  [MATCH_EVENTS.VAR_DECISION]: '📺',
  [MATCH_EVENTS.INJURY]: '🚑'
};

// Injury types and severities
export const INJURY_TYPES = {
  MUSCLE: 'muscle',
  LIGAMENT: 'ligament',
  BONE: 'bone',
  HEAD: 'head',
  KNEE: 'knee',
  ANKLE: 'ankle',
  SHOULDER: 'shoulder',
  BACK: 'back',
  OTHER: 'other'
};

export const INJURY_TYPE_LABELS = {
  [INJURY_TYPES.MUSCLE]: 'Muscle Injury',
  [INJURY_TYPES.LIGAMENT]: 'Ligament Injury',
  [INJURY_TYPES.BONE]: 'Bone Fracture',
  [INJURY_TYPES.HEAD]: 'Head Injury',
  [INJURY_TYPES.KNEE]: 'Knee Injury',
  [INJURY_TYPES.ANKLE]: 'Ankle Injury',
  [INJURY_TYPES.SHOULDER]: 'Shoulder Injury',
  [INJURY_TYPES.BACK]: 'Back Injury',
  [INJURY_TYPES.OTHER]: 'Other Injury'
};

export const INJURY_SEVERITIES = {
  MINOR: 'minor',
  MODERATE: 'moderate',
  MAJOR: 'major',
  CAREER_ENDING: 'career_ending'
};

export const INJURY_SEVERITY_LABELS = {
  [INJURY_SEVERITIES.MINOR]: 'Minor (1-2 weeks)',
  [INJURY_SEVERITIES.MODERATE]: 'Moderate (3-8 weeks)',
  [INJURY_SEVERITIES.MAJOR]: 'Major (2+ months)',
  [INJURY_SEVERITIES.CAREER_ENDING]: 'Career Ending'
};

export const INJURY_SEVERITY_COLORS = {
  [INJURY_SEVERITIES.MINOR]: '#fbbf24',      // Yellow
  [INJURY_SEVERITIES.MODERATE]: '#fb923c',   // Orange
  [INJURY_SEVERITIES.MAJOR]: '#ef4444',      // Red
  [INJURY_SEVERITIES.CAREER_ENDING]: '#7c3aed' // Purple
};

export const INJURY_STATUSES = {
  ACTIVE: 'active',
  RECOVERED: 'recovered',
  PENDING: 'pending'
};

export const INJURY_STATUS_LABELS = {
  [INJURY_STATUSES.ACTIVE]: 'Active',
  [INJURY_STATUSES.RECOVERED]: 'Recovered',
  [INJURY_STATUSES.PENDING]: 'Pending Assessment'
};

// =============================================================================
// UI/UX CONSTANTS
// =============================================================================

// Theme options
export const THEMES = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system'
};

export const THEME_LABELS = {
  [THEMES.LIGHT]: 'Light',
  [THEMES.DARK]: 'Dark',
  [THEMES.SYSTEM]: 'System'
};

// Languages
export const LANGUAGES = {
  EN: 'en',
  ES: 'es',
  FR: 'fr',
  DE: 'de',
  IT: 'it',
  PT: 'pt'
};

export const LANGUAGE_LABELS = {
  [LANGUAGES.EN]: 'English',
  [LANGUAGES.ES]: 'Español',
  [LANGUAGES.FR]: 'Français',
  [LANGUAGES.DE]: 'Deutsch',
  [LANGUAGES.IT]: 'Italiano',
  [LANGUAGES.PT]: 'Português'
};

// Currencies
export const CURRENCIES = {
  EUR: 'EUR',
  USD: 'USD',
  GBP: 'GBP',
  JPY: 'JPY',
  CAD: 'CAD',
  AUD: 'AUD'
};

export const CURRENCY_SYMBOLS = {
  [CURRENCIES.EUR]: '€',
  [CURRENCIES.USD]: '$',
  [CURRENCIES.GBP]: '£',
  [CURRENCIES.JPY]: '¥',
  [CURRENCIES.CAD]: 'C$',
  [CURRENCIES.AUD]: 'A$'
};

// View modes
export const VIEW_MODES = {
  CARDS: 'cards',
  TABLE: 'table',
  LIST: 'list'
};

export const VIEW_MODE_LABELS = {
  [VIEW_MODES.CARDS]: 'Cards View',
  [VIEW_MODES.TABLE]: 'Table View',
  [VIEW_MODES.LIST]: 'List View'
};

// Sort orders
export const SORT_ORDERS = {
  ASC: 'asc',
  DESC: 'desc'
};

export const SORT_ORDER_LABELS = {
  [SORT_ORDERS.ASC]: 'Ascending',
  [SORT_ORDERS.DESC]: 'Descending'
};

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

export const NOTIFICATION_COLORS = {
  [NOTIFICATION_TYPES.SUCCESS]: '#10b981',
  [NOTIFICATION_TYPES.ERROR]: '#ef4444',
  [NOTIFICATION_TYPES.WARNING]: '#f59e0b',
  [NOTIFICATION_TYPES.INFO]: '#3b82f6'
};

// Loading types
export const LOADING_TYPES = {
  GLOBAL: 'global',
  MATCHES: 'matches',
  PLAYERS: 'players',
  TEAMS: 'teams',
  LEAGUES: 'leagues',
  PREDICTIONS: 'predictions',
  INJURIES: 'injuries',
  STATISTICS: 'statistics',
  LOCAL: 'local'
};

// Error types
export const ERROR_TYPES = {
  GLOBAL: 'global',
  API: 'api',
  NETWORK: 'network',
  VALIDATION: 'validation',
  AUTH: 'auth',
  LOCAL: 'local'
};

// =============================================================================
// FILTER AND SEARCH CONSTANTS
// =============================================================================

// Time frames
export const TIME_FRAMES = {
  TODAY: 'today',
  TOMORROW: 'tomorrow',
  WEEK: 'week',
  MONTH: 'month',
  SEASON: 'season',
  ALL: 'all'
};

export const TIME_FRAME_LABELS = {
  [TIME_FRAMES.TODAY]: 'Today',
  [TIME_FRAMES.TOMORROW]: 'Tomorrow',
  [TIME_FRAMES.WEEK]: 'This Week',
  [TIME_FRAMES.MONTH]: 'This Month',
  [TIME_FRAMES.SEASON]: 'This Season',
  [TIME_FRAMES.ALL]: 'All Time'
};

// Division levels
export const DIVISION_LEVELS = {
  FIRST: '1st',
  SECOND: '2nd',
  THIRD: '3rd',
  FOURTH: '4th',
  ALL: 'all'
};

export const DIVISION_LEVEL_LABELS = {
  [DIVISION_LEVELS.FIRST]: '1st Division',
  [DIVISION_LEVELS.SECOND]: '2nd Division',
  [DIVISION_LEVELS.THIRD]: '3rd Division',
  [DIVISION_LEVELS.FOURTH]: '4th Division',
  [DIVISION_LEVELS.ALL]: 'All Divisions'
};

// Availability statuses
export const AVAILABILITY_STATUSES = {
  AVAILABLE: 'available',
  INJURED: 'injured',
  SUSPENDED: 'suspended',
  ALL: 'all'
};

export const AVAILABILITY_STATUS_LABELS = {
  [AVAILABILITY_STATUSES.AVAILABLE]: 'Available',
  [AVAILABILITY_STATUSES.INJURED]: 'Injured',
  [AVAILABILITY_STATUSES.SUSPENDED]: 'Suspended',
  [AVAILABILITY_STATUSES.ALL]: 'All Players'
};

// =============================================================================
// PAGINATION AND LIMITS
// =============================================================================

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
  PAGE_SIZE_OPTIONS: [10, 20, 50, 100],
  MAX_PAGES_DISPLAYED: 7
};

export const LIMITS = {
  SEARCH_HISTORY: 50,
  MESSAGE_HISTORY: 1000,
  MESSAGE_QUEUE: 100,
  FAVORITES_PER_TYPE: 500,
  CACHE_SIZE: 1000,
  MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
  MAX_UPLOAD_FILES: 10
};

// =============================================================================
// TIME AND DATE CONSTANTS
// =============================================================================

export const TIME_ZONES = {
  UTC: 'UTC',
  AMERICA_BOGOTA: 'America/Bogota',
  AMERICA_NEW_YORK: 'America/New_York',
  EUROPE_LONDON: 'Europe/London',
  EUROPE_MADRID: 'Europe/Madrid',
  ASIA_TOKYO: 'Asia/Tokyo'
};

export const TIME_ZONE_LABELS = {
  [TIME_ZONES.UTC]: 'UTC',
  [TIME_ZONES.AMERICA_BOGOTA]: 'Bogotá (COT)',
  [TIME_ZONES.AMERICA_NEW_YORK]: 'New York (EST/EDT)',
  [TIME_ZONES.EUROPE_LONDON]: 'London (GMT/BST)',
  [TIME_ZONES.EUROPE_MADRID]: 'Madrid (CET/CEST)',
  [TIME_ZONES.ASIA_TOKYO]: 'Tokyo (JST)'
};

export const DATE_FORMATS = {
  SHORT: 'MM/dd/yyyy',
  MEDIUM: 'MMM dd, yyyy',
  LONG: 'MMMM dd, yyyy',
  FULL: 'EEEE, MMMM dd, yyyy',
  ISO: 'yyyy-MM-dd',
  TIME: 'HH:mm',
  DATETIME: 'MM/dd/yyyy HH:mm',
  TIMESTAMP: 'yyyy-MM-dd HH:mm:ss'
};

export const REFRESH_INTERVALS = {
  NEVER: 0,
  FAST: 15000,      // 15 seconds
  NORMAL: 30000,    // 30 seconds
  SLOW: 60000,      // 1 minute
  VERY_SLOW: 300000 // 5 minutes
};

// =============================================================================
// USER ROLES AND PERMISSIONS
// =============================================================================

export const USER_ROLES = {
  ADMIN: 'admin',
  MODERATOR: 'moderator',
  PREMIUM: 'premium',
  USER: 'user',
  GUEST: 'guest'
};

export const USER_ROLE_LABELS = {
  [USER_ROLES.ADMIN]: 'Administrator',
  [USER_ROLES.MODERATOR]: 'Moderator',
  [USER_ROLES.PREMIUM]: 'Premium User',
  [USER_ROLES.USER]: 'Standard User',
  [USER_ROLES.GUEST]: 'Guest'
};

export const SUBSCRIPTION_TYPES = {
  FREE: 'free',
  PREMIUM: 'premium',
  PRO: 'pro'
};

export const SUBSCRIPTION_TYPE_LABELS = {
  [SUBSCRIPTION_TYPES.FREE]: 'Free',
  [SUBSCRIPTION_TYPES.PREMIUM]: 'Premium',
  [SUBSCRIPTION_TYPES.PRO]: 'Professional'
};

// =============================================================================
// API AND WEBSOCKET CONSTANTS
// =============================================================================

export const HTTP_STATUS_CODES = {
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

export const REQUEST_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  PATCH: 'PATCH',
  DELETE: 'DELETE'
};

export const CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  URL_ENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  HTML: 'text/html'
};

export const WS_CONNECTION_STATES = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3
};

export const WS_CLOSE_CODES = {
  NORMAL: 1000,
  GOING_AWAY: 1001,
  PROTOCOL_ERROR: 1002,
  UNSUPPORTED_DATA: 1003,
  NO_STATUS: 1005,
  ABNORMAL: 1006,
  INVALID_DATA: 1007,
  POLICY_VIOLATION: 1008,
  MESSAGE_TOO_BIG: 1009,
  EXTENSION_REQUIRED: 1010,
  INTERNAL_ERROR: 1011,
  SERVICE_RESTART: 1012,
  TRY_AGAIN_LATER: 1013,
  BAD_GATEWAY: 1014,
  TLS_HANDSHAKE: 1015
};

// =============================================================================
// PREDICTION AND ANALYTICS CONSTANTS
// =============================================================================

export const PREDICTION_OUTCOMES = {
  HOME_WIN: 'home_win',
  DRAW: 'draw',
  AWAY_WIN: 'away_win'
};

export const PREDICTION_OUTCOME_LABELS = {
  [PREDICTION_OUTCOMES.HOME_WIN]: 'Home Win',
  [PREDICTION_OUTCOMES.DRAW]: 'Draw',
  [PREDICTION_OUTCOMES.AWAY_WIN]: 'Away Win'
};

export const PREDICTION_CONFIDENCE_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high'
};

export const PREDICTION_CONFIDENCE_LABELS = {
  [PREDICTION_CONFIDENCE_LEVELS.LOW]: 'Low Confidence',
  [PREDICTION_CONFIDENCE_LEVELS.MEDIUM]: 'Medium Confidence',
  [PREDICTION_CONFIDENCE_LEVELS.HIGH]: 'High Confidence'
};

export const PREDICTION_CONFIDENCE_COLORS = {
  [PREDICTION_CONFIDENCE_LEVELS.LOW]: '#ef4444',     // Red
  [PREDICTION_CONFIDENCE_LEVELS.MEDIUM]: '#f59e0b',  // Yellow
  [PREDICTION_CONFIDENCE_LEVELS.HIGH]: '#10b981'     // Green
};

export const PREDICTION_CONFIDENCE_THRESHOLDS = {
  [PREDICTION_CONFIDENCE_LEVELS.LOW]: 60,
  [PREDICTION_CONFIDENCE_LEVELS.MEDIUM]: 80,
  [PREDICTION_CONFIDENCE_LEVELS.HIGH]: 100
};

// =============================================================================
// STATISTICS CONSTANTS
// =============================================================================

export const STATISTIC_TYPES = {
  GOALS: 'goals',
  ASSISTS: 'assists',
  APPEARANCES: 'appearances',
  MINUTES: 'minutes',
  YELLOW_CARDS: 'yellow_cards',
  RED_CARDS: 'red_cards',
  CLEAN_SHEETS: 'clean_sheets',
  SAVES: 'saves',
  TACKLES: 'tackles',
  INTERCEPTIONS: 'interceptions',
  PASSES: 'passes',
  PASS_ACCURACY: 'pass_accuracy',
  SHOTS: 'shots',
  SHOTS_ON_TARGET: 'shots_on_target',
  DRIBBLES: 'dribbles',
  CROSSES: 'crosses'
};

export const STATISTIC_LABELS = {
  [STATISTIC_TYPES.GOALS]: 'Goals',
  [STATISTIC_TYPES.ASSISTS]: 'Assists',
  [STATISTIC_TYPES.APPEARANCES]: 'Appearances',
  [STATISTIC_TYPES.MINUTES]: 'Minutes Played',
  [STATISTIC_TYPES.YELLOW_CARDS]: 'Yellow Cards',
  [STATISTIC_TYPES.RED_CARDS]: 'Red Cards',
  [STATISTIC_TYPES.CLEAN_SHEETS]: 'Clean Sheets',
  [STATISTIC_TYPES.SAVES]: 'Saves',
  [STATISTIC_TYPES.TACKLES]: 'Tackles',
  [STATISTIC_TYPES.INTERCEPTIONS]: 'Interceptions',
  [STATISTIC_TYPES.PASSES]: 'Passes',
  [STATISTIC_TYPES.PASS_ACCURACY]: 'Pass Accuracy %',
  [STATISTIC_TYPES.SHOTS]: 'Shots',
  [STATISTIC_TYPES.SHOTS_ON_TARGET]: 'Shots on Target',
  [STATISTIC_TYPES.DRIBBLES]: 'Dribbles',
  [STATISTIC_TYPES.CROSSES]: 'Crosses'
};

// =============================================================================
// VALIDATION CONSTANTS
// =============================================================================

export const VALIDATION_RULES = {
  PASSWORD: {
    MIN_LENGTH: 8,
    MAX_LENGTH: 128,
    REQUIRE_UPPERCASE: true,
    REQUIRE_LOWERCASE: true,
    REQUIRE_NUMBERS: true,
    REQUIRE_SPECIAL_CHARS: true
  },
  EMAIL: {
    MAX_LENGTH: 254,
    PATTERN: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  },
  NAME: {
    MIN_LENGTH: 2,
    MAX_LENGTH: 50,
    PATTERN: /^[a-zA-ZÀ-ÿ\s'-]+$/
  },
  PHONE: {
    PATTERN: /^\+?[\d\s\-\(\)]+$/,
    MIN_LENGTH: 10,
    MAX_LENGTH: 20
  }
};

export const INPUT_LIMITS = {
  SEARCH_QUERY: 100,
  COMMENT: 500,
  DESCRIPTION: 1000,
  BIO: 300,
  TEAM_NAME: 100,
  PLAYER_NAME: 100,
  LEAGUE_NAME: 100
};

// =============================================================================
// FEATURE FLAGS
// =============================================================================

export const FEATURE_FLAGS = {
  ENABLE_DARK_MODE: true,
  ENABLE_NOTIFICATIONS: true,
  ENABLE_REAL_TIME: true,
  ENABLE_PREDICTIONS: true,
  ENABLE_ANALYTICS: true,
  ENABLE_EXPORT: true,
  ENABLE_FAVORITES: true,
  ENABLE_COMMENTS: false,
  ENABLE_SOCIAL_SHARING: false,
  ENABLE_MOBILE_APP: false
};

// =============================================================================
// LOCAL STORAGE KEYS
// =============================================================================

export const STORAGE_KEYS = {
  // Authentication
  AUTH_TOKEN: 'football-analytics-token',
  REFRESH_TOKEN: 'football-analytics-refresh-token',
  USER_DATA: 'football-analytics-user',
  SESSION_DATA: 'football-analytics-session',
  
  // App Settings
  THEME: 'football-analytics-theme',
  LANGUAGE: 'football-analytics-language',
  TIMEZONE: 'football-analytics-timezone',
  CURRENCY: 'football-analytics-currency',
  
  // User Preferences
  PREFERENCES: 'football-analytics-preferences',
  FAVORITES: 'football-analytics-favorites',
  FILTERS: 'football-analytics-filters',
  SEARCH_HISTORY: 'football-analytics-search-history',
  
  // App Configuration
  CONFIG: 'football-analytics-config',
  LAYOUT_SETTINGS: 'football-analytics-layout',
  NOTIFICATION_SETTINGS: 'football-analytics-notifications',
  
  // Cache
  API_CACHE: 'football-analytics-api-cache',
  LEAGUES_CACHE: 'football-analytics-leagues-cache',
  TEAMS_CACHE: 'football-analytics-teams-cache',
  COUNTRIES_CACHE: 'football-analytics-countries-cache',
  
  // Analytics
  ANALYTICS: 'football-analytics-analytics',
  FEATURE_USAGE: 'football-analytics-feature-usage',
  PERFORMANCE_METRICS: 'football-analytics-performance'
};

// =============================================================================
// REGEX PATTERNS
// =============================================================================

export const REGEX_PATTERNS = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PASSWORD: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>])[A-Za-z\d!@#$%^&*(),.?":{}|<>]{8,}$/,
  PHONE: /^\+?[\d\s\-\(\)]+$/,
  URL: /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/,
  SLUG: /^[a-z0-9]+(?:-[a-z0-9]+)*$/,
  HEX_COLOR: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
  UUID: /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  CREDIT_CARD: /^\d{4}\s?\d{4}\s?\d{4}\s?\d{4}$/,
  ZIP_CODE: /^\d{5}(-\d{4})?$/
};

// =============================================================================
// ICONS AND EMOJIS
// =============================================================================

export const ICONS = {
  // General
  LOADING: '⏳',
  SUCCESS: '✅',
  ERROR: '❌',
  WARNING: '⚠️',
  INFO: 'ℹ️',
  
  // Sports
  FOOTBALL: '⚽',
  GOAL: '🥅',
  TROPHY: '🏆',
  MEDAL: '🏅',
  STADIUM: '🏟️',
  
  // Cards and Events
  YELLOW_CARD: '🟨',
  RED_CARD: '🟥',
  SUBSTITUTION: '🔄',
  INJURY: '🚑',
  VAR: '📺',
  
  // Actions
  PLAY: '▶️',
  PAUSE: '⏸️',
  STOP: '⏹️',
  REFRESH: '🔄',
  SEARCH: '🔍',
  FILTER: '🔽',
  STAR: '⭐',
  HEART: '❤️',
  
  // Navigation
  HOME: '🏠',
  BACK: '⬅️',
  FORWARD: '➡️',
  UP: '⬆️',
  DOWN: '⬇️',
  
  // User
  USER: '👤',
  USERS: '👥',
  ADMIN: '👑',
  GUEST: '👻'
};

// =============================================================================
// BREAKPOINTS AND RESPONSIVE
// =============================================================================

export const BREAKPOINTS = {
  XS: 320,
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  XXL: 1536
};

export const DEVICE_TYPES = {
  MOBILE: 'mobile',
  TABLET: 'tablet',
  DESKTOP: 'desktop'
};

// =============================================================================
// ANALYTICS EVENTS
// =============================================================================

export const ANALYTICS_EVENTS = {
  PAGE_VIEW: 'page_view',
  USER_LOGIN: 'user_login',
  USER_LOGOUT: 'user_logout',
  USER_REGISTER: 'user_register',
  SEARCH: 'search',
  FILTER_APPLIED: 'filter_applied',
  FAVORITE_ADDED: 'favorite_added',
  FAVORITE_REMOVED: 'favorite_removed',
  PREDICTION_VIEWED: 'prediction_viewed',
  MATCH_VIEWED: 'match_viewed',
  PLAYER_VIEWED: 'player_viewed',
  TEAM_VIEWED: 'team_viewed',
  EXPORT_DATA: 'export_data',
  ERROR_OCCURRED: 'error_occurred'
};

// =============================================================================
// SOCIAL MEDIA
// =============================================================================

export const SOCIAL_PLATFORMS = {
  FACEBOOK: 'facebook',
  TWITTER: 'twitter',
  INSTAGRAM: 'instagram',
  LINKEDIN: 'linkedin',
  YOUTUBE: 'youtube',
  TIKTOK: 'tiktok'
};

export const SOCIAL_PLATFORM_URLS = {
  [SOCIAL_PLATFORMS.FACEBOOK]: 'https://facebook.com/footballanalytics',
  [SOCIAL_PLATFORMS.TWITTER]: 'https://twitter.com/footballanalytics',
  [SOCIAL_PLATFORMS.INSTAGRAM]: 'https://instagram.com/footballanalytics',
  [SOCIAL_PLATFORMS.LINKEDIN]: 'https://linkedin.com/company/footballanalytics',
  [SOCIAL_PLATFORMS.YOUTUBE]: 'https://youtube.com/footballanalytics',
  [SOCIAL_PLATFORMS.TIKTOK]: 'https://tiktok.com/@footballanalytics'
};

// =============================================================================
// DEFAULT VALUES
// =============================================================================

export const DEFAULTS = {
  THEME: THEMES.LIGHT,
  LANGUAGE: LANGUAGES.EN,
  CURRENCY: CURRENCIES.EUR,
  TIMEZONE: TIME_ZONES.AMERICA_BOGOTA,
  VIEW_MODE: VIEW_MODES.CARDS,
  PAGE_SIZE: PAGINATION.DEFAULT_PAGE_SIZE,
  REFRESH_INTERVAL: REFRESH_INTERVALS.NORMAL,
  CONFEDERATION: CONFEDERATIONS.UEFA,
  TIME_FRAME: TIME_FRAMES.TODAY,
  SORT_ORDER: SORT_ORDERS.DESC,
  NOTIFICATION_TYPE: NOTIFICATION_TYPES.INFO,
  LOADING_TYPE: LOADING_TYPES.LOCAL,
  ERROR_TYPE: ERROR_TYPES.LOCAL
};

// =============================================================================
// EXPORT ALL CONSTANTS
// =============================================================================

export default {
  APP_CONFIG,
  CONFEDERATIONS,
  CONFEDERATION_LABELS,
  CONFEDERATION_COLORS,
  POSITIONS,
  POSITION_LABELS,
  POSITION_COLORS,
  DETAILED_POSITIONS,
  MATCH_STATUSES,
  MATCH_STATUS_LABELS,
  MATCH_STATUS_COLORS,
  MATCH_EVENTS,
  MATCH_EVENT_ICONS,
  INJURY_TYPES,
  INJURY_TYPE_LABELS,
  INJURY_SEVERITIES,
  INJURY_SEVERITY_LABELS,
  INJURY_SEVERITY_COLORS,
  INJURY_STATUSES,
  INJURY_STATUS_LABELS,
  THEMES,
  THEME_LABELS,
  LANGUAGES,
  LANGUAGE_LABELS,
  CURRENCIES,
  CURRENCY_SYMBOLS,
  VIEW_MODES,
  VIEW_MODE_LABELS,
  SORT_ORDERS,
  SORT_ORDER_LABELS,
  NOTIFICATION_TYPES,
  NOTIFICATION_COLORS,
  LOADING_TYPES,
  ERROR_TYPES,
  TIME_FRAMES,
  TIME_FRAME_LABELS,
  DIVISION_LEVELS,
  DIVISION_LEVEL_LABELS,
  AVAILABILITY_STATUSES,
  AVAILABILITY_STATUS_LABELS,
  PAGINATION,
  LIMITS,
  TIME_ZONES,
  TIME_ZONE_LABELS,
  DATE_FORMATS,
  REFRESH_INTERVALS,
  USER_ROLES,
  USER_ROLE_LABELS,
  SUBSCRIPTION_TYPES,
  SUBSCRIPTION_TYPE_LABELS,
  HTTP_STATUS_CODES,
  REQUEST_METHODS,
  CONTENT_TYPES,
  WS_CONNECTION_STATES,
  WS_CLOSE_CODES,
  PREDICTION_OUTCOMES,
  PREDICTION_OUTCOME_LABELS,
  PREDICTION_CONFIDENCE_LEVELS,
  PREDICTION_CONFIDENCE_LABELS,
  PREDICTION_CONFIDENCE_COLORS,
  PREDICTION_CONFIDENCE_THRESHOLDS,
  STATISTIC_TYPES,
  STATISTIC_LABELS,
  VALIDATION_RULES,
  INPUT_LIMITS,
  FEATURE_FLAGS,
  STORAGE_KEYS,
  REGEX_PATTERNS,
  ICONS,
  BREAKPOINTS,
  DEVICE_TYPES,
  ANALYTICS_EVENTS,
  SOCIAL_PLATFORMS,
  SOCIAL_PLATFORM_URLS,
  DEFAULTS
};