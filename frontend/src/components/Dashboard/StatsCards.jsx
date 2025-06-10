import {
  Activity,
  AlertCircle,
  BarChart3,
  Brain,
  Calendar,
  CheckCircle,
  Minus,
  RefreshCw,
  Star,
  Target,
  TrendingDown,
  TrendingUp,
  Trophy,
  Users
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// Import CSS
import './Dashboard.css';

const StatsCards = ({ 
  stats = [], 
  loading = false, 
  error = null,
  className = '',
  onCardClick = null,
  showTrends = true,
  cardSize = 'medium', // small, medium, large
  columns = 'auto' // auto, 1, 2, 3, 4
}) => {
  const navigate = useNavigate();

  // Handle card click
  const handleCardClick = (stat) => {
    if (onCardClick) {
      onCardClick(stat);
    } else if (stat.path) {
      navigate(stat.path);
    } else {
      // Default navigation based on stat type
      switch (stat.type) {
        case 'leagues':
          navigate('/leagues');
          break;
        case 'predictions':
          navigate('/predictions');
          break;
        case 'accuracy':
          navigate('/analytics/charts');
          break;
        case 'live':
          navigate('/live');
          break;
        case 'teams':
          navigate('/teams');
          break;
        case 'analytics':
          navigate('/analytics');
          break;
        default:
          break;
      }
    }
  };

  // Get grid class based on columns prop
  const getGridClass = () => {
    if (columns === 'auto') return 'dashboard-metrics-grid';
    return `grid grid-cols-1 md:grid-cols-${Math.min(columns, 4)} gap-6`;
  };

  // Get card size class
  const getSizeClass = () => {
    switch (cardSize) {
      case 'small':
        return 'p-4';
      case 'large':
        return 'p-8';
      default:
        return 'p-6';
    }
  };

  // Format number with appropriate suffixes
  const formatNumber = (value) => {
    if (typeof value === 'string') return value;
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toLocaleString();
  };

  // Get trend icon and class
  const getTrendIndicator = (change, changeType) => {
    if (!showTrends || change === 0) return null;
    
    const absChange = Math.abs(change);
    let IconComponent, colorClass;
    
    switch (changeType) {
      case 'positive':
        IconComponent = TrendingUp;
        colorClass = 'text-green-600 dark:text-green-400';
        break;
      case 'negative':
        IconComponent = TrendingDown;
        colorClass = 'text-red-600 dark:text-red-400';
        break;
      default:
        IconComponent = Minus;
        colorClass = 'text-gray-500 dark:text-gray-400';
    }

    return (
      <div className={`flex items-center gap-1 text-xs font-medium ${colorClass}`}>
        <IconComponent size={12} />
        <span>{absChange.toFixed(1)}%</span>
      </div>
    );
  };

  // Get icon color based on stat type
  const getIconColor = (type, variant = 'default') => {
    const colors = {
      leagues: variant === 'bg' ? 'bg-orange-100 dark:bg-orange-900' : 'text-orange-600 dark:text-orange-400',
      predictions: variant === 'bg' ? 'bg-green-100 dark:bg-green-900' : 'text-green-600 dark:text-green-400',
      accuracy: variant === 'bg' ? 'bg-purple-100 dark:bg-purple-900' : 'text-purple-600 dark:text-purple-400',
      live: variant === 'bg' ? 'bg-red-100 dark:bg-red-900' : 'text-red-600 dark:text-red-400',
      teams: variant === 'bg' ? 'bg-blue-100 dark:bg-blue-900' : 'text-blue-600 dark:text-blue-400',
      analytics: variant === 'bg' ? 'bg-indigo-100 dark:bg-indigo-900' : 'text-indigo-600 dark:text-indigo-400',
      players: variant === 'bg' ? 'bg-emerald-100 dark:bg-emerald-900' : 'text-emerald-600 dark:text-emerald-400',
      matches: variant === 'bg' ? 'bg-yellow-100 dark:bg-yellow-900' : 'text-yellow-600 dark:text-yellow-400'
    };
    return colors[type] || (variant === 'bg' ? 'bg-gray-100 dark:bg-gray-800' : 'text-gray-600 dark:text-gray-400');
  };

  // Loading skeleton
  if (loading) {
    return (
      <div className={`${getGridClass()} ${className}`}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className={`metric-card animate-pulse ${getSizeClass()}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
              <div className="w-16 h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
            </div>
            <div className="w-20 h-8 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
            <div className="w-24 h-4 bg-gray-200 dark:bg-gray-700 rounded mb-3"></div>
            <div className="flex justify-between items-center">
              <div className="w-20 h-3 bg-gray-200 dark:bg-gray-700 rounded"></div>
              <div className="w-12 h-3 bg-gray-200 dark:bg-gray-700 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`${getGridClass()} ${className}`}>
        <div className="col-span-full">
          <div className="metric-card border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20">
            <div className="flex items-center gap-3 text-red-700 dark:text-red-400">
              <AlertCircle size={24} />
              <div>
                <h3 className="font-semibold">Error Loading Stats</h3>
                <p className="text-sm opacity-80">{error}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${getGridClass()} ${className}`}>
      {stats.map((stat, index) => {
        const IconComponent = stat.icon || BarChart3;
        const isClickable = stat.clickable !== false && (onCardClick || stat.path || stat.type);
        
        return (
          <div
            key={stat.id || index}
            onClick={isClickable ? () => handleCardClick(stat) : undefined}
            className={`
              metric-card 
              ${getSizeClass()}
              ${isClickable ? 'cursor-pointer hover:scale-105' : 'cursor-default'}
              ${stat.highlight ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
              ${stat.isLive ? 'pulse-glow' : ''}
              ${stat.variant === 'gradient' ? 'bg-gradient-to-br from-blue-600 to-green-600 text-white' : ''}
              transition-all duration-300
            `}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className={`
                w-12 h-12 rounded-lg flex items-center justify-center
                ${stat.variant === 'gradient' ? 'bg-white/20' : getIconColor(stat.type, 'bg')}
              `}>
                <IconComponent 
                  size={24} 
                  className={stat.variant === 'gradient' ? 'text-white' : getIconColor(stat.type)}
                />
              </div>
              
              <div className="flex items-center gap-2">
                {/* Status indicators */}
                {stat.status === 'up' && <CheckCircle size={16} className="text-green-500" />}
                {stat.status === 'down' && <AlertCircle size={16} className="text-red-500" />}
                {stat.status === 'loading' && <RefreshCw size={16} className="text-blue-500 animate-spin" />}
                
                {/* Live indicator */}
                {stat.isLive && (
                  <div className="live-indicator">
                    LIVE
                  </div>
                )}
                
                {/* New badge */}
                {stat.isNew && (
                  <div className="dashboard-badge dashboard-badge--new">
                    NEW
                  </div>
                )}
              </div>
            </div>

            {/* Value */}
            <div className={`
              text-3xl font-bold mb-2
              ${stat.variant === 'gradient' ? 'text-white' : 'metric-value'}
            `}>
              {typeof stat.value === 'number' ? formatNumber(stat.value) : stat.value}
              {stat.suffix && (
                <span className="text-lg font-medium opacity-75">{stat.suffix}</span>
              )}
            </div>

            {/* Label */}
            <div className={`
              text-sm font-medium mb-3
              ${stat.variant === 'gradient' ? 'text-white/90' : 'text-gray-700 dark:text-gray-300'}
            `}>
              {stat.label}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between">
              <div className={`
                text-xs
                ${stat.variant === 'gradient' ? 'text-white/75' : 'text-gray-500 dark:text-gray-400'}
              `}>
                {stat.description || stat.subtitle}
              </div>
              
              {/* Trend indicator */}
              {showTrends && stat.change !== undefined && (
                <div className={stat.variant === 'gradient' ? 'text-white/90' : ''}>
                  {getTrendIndicator(stat.change, stat.changeType)}
                </div>
              )}
            </div>

            {/* Progress bar (optional) */}
            {stat.progress !== undefined && (
              <div className="mt-3">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      stat.variant === 'gradient' 
                        ? 'bg-white/60' 
                        : `bg-${stat.progressColor || 'blue'}-500`
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, stat.progress))}%` }}
                  ></div>
                </div>
                <div className={`
                  text-xs mt-1 text-right
                  ${stat.variant === 'gradient' ? 'text-white/75' : 'text-gray-500 dark:text-gray-400'}
                `}>
                  {stat.progress.toFixed(1)}%
                </div>
              </div>
            )}

            {/* Sub-metrics (optional) */}
            {stat.subMetrics && stat.subMetrics.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="grid grid-cols-2 gap-3">
                  {stat.subMetrics.map((subMetric, subIndex) => (
                    <div key={subIndex} className="text-center">
                      <div className={`
                        text-lg font-semibold
                        ${stat.variant === 'gradient' ? 'text-white' : 'text-gray-900 dark:text-white'}
                      `}>
                        {typeof subMetric.value === 'number' ? formatNumber(subMetric.value) : subMetric.value}
                      </div>
                      <div className={`
                        text-xs
                        ${stat.variant === 'gradient' ? 'text-white/75' : 'text-gray-500 dark:text-gray-400'}
                      `}>
                        {subMetric.label}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action buttons (optional) */}
            {stat.actions && stat.actions.length > 0 && (
              <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex gap-2">
                  {stat.actions.map((action, actionIndex) => {
                    const ActionIcon = action.icon;
                    return (
                      <button
                        key={actionIndex}
                        onClick={(e) => {
                          e.stopPropagation();
                          action.onClick(stat);
                        }}
                        className={`
                          px-3 py-1 text-xs rounded-md transition-colors flex items-center gap-1
                          ${stat.variant === 'gradient' 
                            ? 'bg-white/20 text-white hover:bg-white/30' 
                            : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                          }
                        `}
                      >
                        {ActionIcon && <ActionIcon size={12} />}
                        {action.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

// Predefined stat configurations for common Football Analytics metrics
export const FOOTBALL_STATS_CONFIGS = {
  leagues: {
    type: 'leagues',
    icon: Trophy,
    label: 'Active Leagues',
    description: 'Worldwide coverage',
    path: '/leagues'
  },
  predictions: {
    type: 'predictions',
    icon: Target,
    label: 'AI Predictions',
    description: 'This week',
    path: '/predictions'
  },
  accuracy: {
    type: 'accuracy',
    icon: Brain,
    label: 'Model Accuracy',
    description: 'Last 30 days',
    path: '/analytics/charts'
  },
  liveMatches: {
    type: 'live',
    icon: Activity,
    label: 'Live Matches',
    description: 'Currently playing',
    path: '/live',
    isLive: true
  },
  teams: {
    type: 'teams',
    icon: Users,
    label: 'Teams',
    description: 'Database',
    path: '/teams'
  },
  totalMatches: {
    type: 'matches',
    icon: Calendar,
    label: 'Total Matches',
    description: 'This season'
  },
  topPredictions: {
    type: 'predictions',
    icon: Star,
    label: 'High Confidence',
    description: 'Top predictions',
    path: '/predictions?confidence=high'
  },
  modelUpdates: {
    type: 'analytics',
    icon: RefreshCw,
    label: 'Model Updates',
    description: 'Last updated'
  }
};

// Helper function to create quick stats
export const createFootballStats = (data) => {
  return [
    {
      ...FOOTBALL_STATS_CONFIGS.leagues,
      value: data.totalLeagues || 265,
      change: data.leaguesChange || 0,
      changeType: 'positive'
    },
    {
      ...FOOTBALL_STATS_CONFIGS.predictions,
      value: data.totalPredictions || 0,
      change: data.predictionsChange || 0,
      changeType: data.predictionsChange >= 0 ? 'positive' : 'negative'
    },
    {
      ...FOOTBALL_STATS_CONFIGS.accuracy,
      value: data.modelAccuracy ? `${(data.modelAccuracy * 100).toFixed(1)}%` : 'N/A',
      change: data.accuracyChange || 0,
      changeType: data.accuracyChange >= 0 ? 'positive' : 'negative'
    },
    {
      ...FOOTBALL_STATS_CONFIGS.liveMatches,
      value: data.liveMatches || 0,
      change: 0,
      changeType: 'neutral'
    }
  ];
};

export default StatsCards;