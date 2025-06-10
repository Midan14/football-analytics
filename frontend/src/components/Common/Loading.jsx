import {
    Activity,
    BarChart3,
    Brain,
    Database,
    RefreshCw,
    Search,
    Target,
    TrendingUp,
    Trophy,
    Users
} from 'lucide-react';

// Componente principal de Loading
const Loading = ({ 
  type = 'spinner', 
  size = 'medium', 
  message = 'Loading...', 
  context = 'general',
  fullscreen = false,
  className = ''
}) => {
  
  // Configuración de tamaños
  const sizes = {
    small: { spinner: 16, container: 'p-4', text: 'text-sm' },
    medium: { spinner: 24, container: 'p-6', text: 'text-base' },
    large: { spinner: 32, container: 'p-8', text: 'text-lg' },
    xl: { spinner: 48, container: 'p-12', text: 'text-xl' }
  };

  // Configuración de contextos con iconos y colores específicos
  const contexts = {
    general: { 
      icon: RefreshCw, 
      color: 'text-blue-600', 
      bg: 'bg-blue-50',
      message: 'Loading Football Analytics...'
    },
    predictions: { 
      icon: Target, 
      color: 'text-green-600', 
      bg: 'bg-green-50',
      message: 'Loading AI predictions...'
    },
    analytics: { 
      icon: BarChart3, 
      color: 'text-purple-600', 
      bg: 'bg-purple-50',
      message: 'Processing analytics data...'
    },
    teams: { 
      icon: Users, 
      color: 'text-blue-600', 
      bg: 'bg-blue-50',
      message: 'Loading team data...'
    },
    leagues: { 
      icon: Trophy, 
      color: 'text-orange-600', 
      bg: 'bg-orange-50',
      message: 'Loading league information...'
    },
    live: { 
      icon: Activity, 
      color: 'text-red-600', 
      bg: 'bg-red-50',
      message: 'Loading live match data...'
    },
    search: { 
      icon: Search, 
      color: 'text-gray-600', 
      bg: 'bg-gray-50',
      message: 'Searching...'
    },
    ai: { 
      icon: Brain, 
      color: 'text-indigo-600', 
      bg: 'bg-indigo-50',
      message: 'AI processing...'
    },
    statistics: { 
      icon: TrendingUp, 
      color: 'text-emerald-600', 
      bg: 'bg-emerald-50',
      message: 'Loading statistics...'
    },
    database: { 
      icon: Database, 
      color: 'text-slate-600', 
      bg: 'bg-slate-50',
      message: 'Fetching data...'
    }
  };

  const currentSize = sizes[size];
  const currentContext = contexts[context] || contexts.general;
  const IconComponent = currentContext.icon;
  const displayMessage = message === 'Loading...' ? currentContext.message : message;

  // Wrapper para fullscreen
  if (fullscreen) {
    return (
      <div className="fixed inset-0 bg-white dark:bg-gray-900 flex items-center justify-center z-50">
        <div className="text-center">
          <div className="mb-6">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-3xl">⚽</span>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Football Analytics
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              {displayMessage}
            </p>
          </div>
          <LoadingSpinner size={currentSize.spinner} color={currentContext.color} />
        </div>
      </div>
    );
  }

  // Renderizar según el tipo
  switch (type) {
    case 'spinner':
      return <LoadingSpinner size={currentSize.spinner} color={currentContext.color} message={displayMessage} className={className} />;
    
    case 'dots':
      return <LoadingDots color={currentContext.color} message={displayMessage} className={className} />;
    
    case 'pulse':
      return <LoadingPulse color={currentContext.color} message={displayMessage} className={className} />;
    
    case 'card':
      return <LoadingCard context={currentContext} size={currentSize} message={displayMessage} className={className} />;
    
    case 'skeleton':
      return <LoadingSkeleton className={className} />;
    
    case 'table':
      return <LoadingTable className={className} />;
    
    case 'chart':
      return <LoadingChart context={currentContext} size={currentSize} message={displayMessage} className={className} />;
    
    default:
      return <LoadingSpinner size={currentSize.spinner} color={currentContext.color} message={displayMessage} className={className} />;
  }
};

// Componente de Spinner básico
const LoadingSpinner = ({ size = 24, color = 'text-blue-600', message, className = '' }) => (
  <div className={`flex flex-col items-center justify-center ${className}`}>
    <RefreshCw 
      size={size} 
      className={`animate-spin ${color} mb-2`} 
    />
    {message && (
      <p className="text-gray-600 dark:text-gray-400 text-sm animate-pulse">
        {message}
      </p>
    )}
  </div>
);

// Componente de Dots
const LoadingDots = ({ color = 'bg-blue-600', message, className = '' }) => (
  <div className={`flex flex-col items-center justify-center ${className}`}>
    <div className="flex space-x-1 mb-3">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`w-2 h-2 ${color} rounded-full animate-bounce`}
          style={{ animationDelay: `${i * 0.1}s` }}
        ></div>
      ))}
    </div>
    {message && (
      <p className="text-gray-600 dark:text-gray-400 text-sm">
        {message}
      </p>
    )}
  </div>
);

// Componente de Pulse
const LoadingPulse = ({ color = 'text-blue-600', message, className = '' }) => (
  <div className={`flex flex-col items-center justify-center ${className}`}>
    <div className="relative">
      <div className={`w-12 h-12 ${color.replace('text-', 'bg-')} rounded-full animate-ping opacity-20`}></div>
      <div className={`absolute inset-0 w-12 h-12 ${color.replace('text-', 'bg-')} rounded-full animate-pulse`}></div>
      <div className="absolute inset-2 w-8 h-8 bg-white dark:bg-gray-900 rounded-full flex items-center justify-center">
        <span className="text-xl">⚽</span>
      </div>
    </div>
    {message && (
      <p className="text-gray-600 dark:text-gray-400 text-sm mt-3">
        {message}
      </p>
    )}
  </div>
);

// Componente de Card
const LoadingCard = ({ context, size, message, className = '' }) => {
  const IconComponent = context.icon;
  
  return (
    <div className={`${context.bg} dark:bg-gray-800 rounded-xl ${size.container} text-center ${className}`}>
      <div className="mb-4">
        <div className={`w-16 h-16 ${context.color.replace('text-', 'bg-')} bg-opacity-10 rounded-full flex items-center justify-center mx-auto mb-3`}>
          <IconComponent size={24} className={`${context.color} animate-pulse`} />
        </div>
        <p className={`${context.color} font-semibold ${size.text}`}>
          {message}
        </p>
      </div>
      <LoadingSpinner size={20} color={context.color} />
    </div>
  );
};

// Componente de Skeleton
const LoadingSkeleton = ({ rows = 5, className = '' }) => (
  <div className={`animate-pulse space-y-4 ${className}`}>
    {/* Header Skeleton */}
    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-lg w-1/3"></div>
    
    {/* Content Skeleton */}
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex space-x-4">
          <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-full flex-shrink-0"></div>
          <div className="flex-1 space-y-2">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          </div>
          <div className="w-16 h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      ))}
    </div>
  </div>
);

// Componente de Table Skeleton
const LoadingTable = ({ rows = 10, columns = 6, className = '' }) => (
  <div className={`animate-pulse ${className}`}>
    {/* Table Header */}
    <div className="grid grid-cols-6 gap-4 p-4 border-b border-gray-200 dark:border-gray-700">
      {Array.from({ length: columns }).map((_, i) => (
        <div key={i} className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
      ))}
    </div>
    
    {/* Table Rows */}
    <div className="space-y-1">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="grid grid-cols-6 gap-4 p-4 hover:bg-gray-50 dark:hover:bg-gray-800">
          {Array.from({ length: columns }).map((_, j) => (
            <div 
              key={j} 
              className={`h-4 bg-gray-200 dark:bg-gray-700 rounded ${j === 0 ? 'w-8' : j === 1 ? 'w-full' : 'w-12'}`}
            ></div>
          ))}
        </div>
      ))}
    </div>
  </div>
);

// Componente de Chart Skeleton
const LoadingChart = ({ context, size, message, className = '' }) => {
  const IconComponent = context.icon;
  
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl ${size.container} ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <IconComponent size={20} className={context.color} />
          <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-32 animate-pulse"></div>
        </div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20 animate-pulse"></div>
      </div>
      
      {/* Chart Area */}
      <div className="h-64 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <BarChart3 size={48} className={`${context.color} opacity-50 animate-pulse mb-3 mx-auto`} />
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            {message}
          </p>
        </div>
      </div>
    </div>
  );
};

// Componentes especializados para casos específicos de Football Analytics

// Loading para Dashboard
export const DashboardLoading = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="bg-white dark:bg-gray-800 rounded-xl p-6 animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
            <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
          </div>
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-16 mb-2"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20"></div>
        </div>
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <LoadingChart context={{ icon: BarChart3, color: 'text-blue-600' }} size={{ container: 'p-6' }} message="Loading chart data..." />
      <LoadingChart context={{ icon: Target, color: 'text-green-600' }} size={{ container: 'p-6' }} message="Loading predictions..." />
    </div>
  </div>
);

// Loading para Predicciones
export const PredictionsLoading = () => (
  <div className="space-y-6">
    <Loading type="card" context="predictions" size="large" message="AI is processing match predictions..." />
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="bg-white dark:bg-gray-800 rounded-xl p-6 animate-pulse">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded flex-1"></div>
            <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {Array.from({ length: 3 }).map((_, j) => (
              <div key={j} className="text-center p-2">
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded mb-1"></div>
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Loading para Live Matches
export const LiveMatchesLoading = () => (
  <div className="space-y-4">
    <div className="flex items-center space-x-3">
      <Activity size={24} className="text-red-600 animate-pulse" />
      <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-32 animate-pulse"></div>
      <div className="w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
    </div>
    <div className="space-y-3">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="bg-white dark:bg-gray-800 rounded-lg p-4 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 flex-1">
              <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-12"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
              <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
            </div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Loading para Estadísticas
export const StatisticsLoading = () => (
  <div className="space-y-6">
    <div className="flex items-center space-x-3">
      <TrendingUp size={24} className="text-emerald-600 animate-pulse" />
      <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-40 animate-pulse"></div>
    </div>
    <LoadingTable rows={15} columns={8} />
  </div>
);

export default Loading;