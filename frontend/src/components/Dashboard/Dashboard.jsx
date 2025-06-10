import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Award,
  BarChart3,
  Brain,
  Clock,
  ExternalLink,
  Globe,
  Minus,
  Pause,
  Play,
  RefreshCw,
  Star,
  Target,
  TrendingUp,
  Trophy,
  Zap
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

// Import components
import { DashboardLoading } from '../Common/Loading';
import './Dashboard.css';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const Dashboard = () => {
  // Estados principales
  const [dashboardData, setDashboardData] = useState(null);
  const [liveMatches, setLiveMatches] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [topLeagues, setTopLeagues] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [predictionTrends, setPredictionTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Estados de configuración
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30000); // 30 segundos
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h'); // 24h, 7d, 30d

  // Hook de navegación
  const navigate = useNavigate();

  // Configuración de colores para gráficos
  const chartColors = {
    primary: '#3B82F6',
    secondary: '#10B981',
    tertiary: '#F59E0B',
    quaternary: '#EF4444',
    quinary: '#8B5CF6',
    senary: '#F97316'
  };

  // Efectos
  useEffect(() => {
    fetchDashboardData();
    
    // Auto-refresh setup
    let interval;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchDashboardData(false); // Silent refresh
      }, refreshInterval);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshInterval, selectedTimeRange]);

  // API Functions
  const fetchDashboardData = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    setError(null);

    try {
      // Fetch all dashboard data in parallel
      const [
        metricsResponse,
        liveMatchesResponse,
        predictionsResponse,
        leaguesResponse,
        performanceResponse,
        trendsResponse
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/dashboard/metrics?timeRange=${selectedTimeRange}`),
        fetch(`${API_BASE_URL}/matches/live`),
        fetch(`${API_BASE_URL}/predictions/recent?limit=10`),
        fetch(`${API_BASE_URL}/leagues/top?limit=8`),
        fetch(`${API_BASE_URL}/models/performance`),
        fetch(`${API_BASE_URL}/dashboard/trends?timeRange=${selectedTimeRange}`)
      ]);

      // Parse responses
      const metrics = metricsResponse.ok ? await metricsResponse.json() : null;
      const live = liveMatchesResponse.ok ? await liveMatchesResponse.json() : null;
      const predictions = predictionsResponse.ok ? await predictionsResponse.json() : null;
      const leagues = leaguesResponse.ok ? await leaguesResponse.json() : null;
      const performance = performanceResponse.ok ? await performanceResponse.json() : null;
      const trends = trendsResponse.ok ? await trendsResponse.json() : null;

      // Update states
      if (metrics) setDashboardData(metrics.data);
      if (live) setLiveMatches(live.matches || []);
      if (predictions) setRecentPredictions(predictions.predictions || []);
      if (leagues) setTopLeagues(leagues.leagues || []);
      if (performance) setModelPerformance(performance.performance);
      if (trends) setPredictionTrends(trends.trends || []);

      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Failed to load dashboard data. Please try again.');
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  // Computed values
  const dashboardMetrics = useMemo(() => {
    if (!dashboardData) return [];

    return [
      {
        id: 'total_leagues',
        label: 'Active Leagues',
        value: dashboardData.total_leagues || 265,
        change: dashboardData.leagues_change || 0,
        changeType: 'positive',
        icon: Trophy,
        color: 'primary',
        description: 'Worldwide coverage'
      },
      {
        id: 'total_predictions',
        label: 'AI Predictions',
        value: dashboardData.total_predictions || 0,
        change: dashboardData.predictions_change || 0,
        changeType: dashboardData.predictions_change >= 0 ? 'positive' : 'negative',
        icon: Target,
        color: 'success',
        description: 'This week'
      },
      {
        id: 'model_accuracy',
        label: 'Model Accuracy',
        value: dashboardData.model_accuracy ? `${(dashboardData.model_accuracy * 100).toFixed(1)}%` : 'N/A',
        change: dashboardData.accuracy_change || 0,
        changeType: dashboardData.accuracy_change >= 0 ? 'positive' : 'negative',
        icon: Brain,
        color: 'info',
        description: 'Last 30 days'
      },
      {
        id: 'live_matches',
        label: 'Live Matches',
        value: liveMatches.length,
        change: 0,
        changeType: 'neutral',
        icon: Activity,
        color: 'danger',
        description: 'Currently playing',
        isLive: true
      }
    ];
  }, [dashboardData, liveMatches]);

  // Chart data preparations
  const accuracyTrendData = useMemo(() => {
    return predictionTrends.map(trend => ({
      date: new Date(trend.date).toLocaleDateString(),
      accuracy: (trend.accuracy * 100).toFixed(1),
      predictions: trend.total_predictions,
      confidence: (trend.avg_confidence * 100).toFixed(1)
    }));
  }, [predictionTrends]);

  const leaguePerformanceData = useMemo(() => {
    return topLeagues.map(league => ({
      name: league.name.length > 15 ? league.name.substring(0, 15) + '...' : league.name,
      accuracy: (league.prediction_accuracy * 100).toFixed(1),
      predictions: league.total_predictions,
      country: league.country
    }));
  }, [topLeagues]);

  const predictionDistributionData = useMemo(() => {
    if (!dashboardData || !dashboardData.prediction_distribution) return [];
    
    return [
      { name: 'Home Win', value: dashboardData.prediction_distribution.home_wins, color: chartColors.secondary },
      { name: 'Draw', value: dashboardData.prediction_distribution.draws, color: chartColors.tertiary },
      { name: 'Away Win', value: dashboardData.prediction_distribution.away_wins, color: chartColors.quaternary }
    ];
  }, [dashboardData]);

  // Event handlers
  const handleMetricClick = (metric) => {
    switch (metric.id) {
      case 'total_leagues':
        navigate('/leagues');
        break;
      case 'total_predictions':
        navigate('/predictions');
        break;
      case 'model_accuracy':
        navigate('/analytics/charts');
        break;
      case 'live_matches':
        navigate('/live');
        break;
      default:
        break;
    }
  };

  const handleRefresh = () => {
    fetchDashboardData(true);
  };

  const formatChange = (change, type) => {
    const absChange = Math.abs(change);
    const icon = type === 'positive' ? ArrowUpRight : type === 'negative' ? ArrowDownRight : Minus;
    const IconComponent = icon;
    
    return (
      <div className={`metric-change metric-change--${type}`}>
        <IconComponent size={12} />
        <span>{absChange.toFixed(1)}%</span>
      </div>
    );
  };

  if (loading) {
    return <DashboardLoading />;
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <div className="p-8 text-center">
          <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg inline-block">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="text-red-600" size={20} />
              <span className="font-semibold">Dashboard Error</span>
            </div>
            <p>{error}</p>
            <button
              onClick={handleRefresh}
              className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2 mx-auto"
            >
              <RefreshCw size={16} />
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-grid">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Football Analytics Dashboard
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time insights from {dashboardData?.total_leagues || 265}+ leagues worldwide
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Time Range Selector */}
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>

            {/* Auto Refresh Toggle */}
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                autoRefresh
                  ? 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900 dark:text-green-300'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300'
              }`}
            >
              {autoRefresh ? <Play size={16} /> : <Pause size={16} />}
              Auto Refresh
            </button>

            {/* Manual Refresh */}
            <button
              onClick={handleRefresh}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </div>

        {/* Last Updated */}
        <div className="text-sm text-gray-500 dark:text-gray-400 mb-6 flex items-center gap-2">
          <Clock size={14} />
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>

        {/* Metrics Grid */}
        <div className="dashboard-metrics-grid">
          {dashboardMetrics.map((metric) => {
            const IconComponent = metric.icon;
            return (
              <div
                key={metric.id}
                onClick={() => handleMetricClick(metric)}
                className={`metric-card metric-card--${metric.color} cursor-pointer ${metric.isLive ? 'pulse-glow' : ''}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className={`w-12 h-12 rounded-lg bg-${metric.color === 'primary' ? 'blue' : metric.color === 'success' ? 'green' : metric.color === 'info' ? 'purple' : 'red'}-100 dark:bg-${metric.color === 'primary' ? 'blue' : metric.color === 'success' ? 'green' : metric.color === 'info' ? 'purple' : 'red'}-900 flex items-center justify-center`}>
                    <IconComponent size={24} className={`text-${metric.color === 'primary' ? 'blue' : metric.color === 'success' ? 'green' : metric.color === 'info' ? 'purple' : 'red'}-600 dark:text-${metric.color === 'primary' ? 'blue' : metric.color === 'success' ? 'green' : metric.color === 'info' ? 'purple' : 'red'}-400`} />
                  </div>
                  {metric.isLive && (
                    <div className="live-indicator">
                      LIVE
                    </div>
                  )}
                </div>
                
                <div className="metric-value">
                  {metric.value}
                </div>
                
                <div className="metric-label">
                  {metric.label}
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {metric.description}
                  </span>
                  {metric.change !== 0 && formatChange(metric.change, metric.changeType)}
                </div>
              </div>
            );
          })}
        </div>

        {/* Charts Grid */}
        <div className="dashboard-charts-grid">
          {/* Model Performance Trend */}
          <div className="chart-container">
            <div className="chart-header">
              <div>
                <h3 className="chart-title">
                  <TrendingUp size={20} />
                  Model Performance Trend
                </h3>
                <p className="chart-subtitle">
                  AI prediction accuracy over time
                </p>
              </div>
              <button
                onClick={() => navigate('/analytics/charts')}
                className="text-blue-600 hover:text-blue-700 dark:text-blue-400"
              >
                <ExternalLink size={16} />
              </button>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={accuracyTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value, name) => [`${value}%`, name === 'accuracy' ? 'Accuracy' : 'Confidence']}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="accuracy"
                    stroke={chartColors.primary}
                    fill={chartColors.primary}
                    fillOpacity={0.1}
                    strokeWidth={2}
                    name="Accuracy"
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence"
                    stroke={chartColors.secondary}
                    fill={chartColors.secondary}
                    fillOpacity={0.1}
                    strokeWidth={2}
                    name="Confidence"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* League Performance */}
          <div className="chart-container">
            <div className="chart-header">
              <div>
                <h3 className="chart-title">
                  <Trophy size={20} />
                  Top Performing Leagues
                </h3>
                <p className="chart-subtitle">
                  Prediction accuracy by league
                </p>
              </div>
              <button
                onClick={() => navigate('/analytics/statistics?tab=leagues')}
                className="text-blue-600 hover:text-blue-700 dark:text-blue-400"
              >
                <ExternalLink size={16} />
              </button>
            </div>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={leaguePerformanceData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis type="category" dataKey="name" width={120} />
                  <Tooltip 
                    formatter={(value) => [`${value}%`, 'Accuracy']}
                  />
                  <Bar dataKey="accuracy" fill={chartColors.secondary} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Content Grid */}
        <div className="dashboard-content-grid">
          {/* Main Content */}
          <div className="space-y-6">
            {/* Prediction Distribution */}
            <div className="chart-container">
              <div className="chart-header">
                <div>
                  <h3 className="chart-title">
                    <Target size={20} />
                    Prediction Distribution
                  </h3>
                  <p className="chart-subtitle">
                    Breakdown of prediction outcomes
                  </p>
                </div>
              </div>
              
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={predictionDistributionData}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {predictionDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Predictions */}
            <div className="chart-container">
              <div className="chart-header">
                <div>
                  <h3 className="chart-title">
                    <Brain size={20} />
                    Recent AI Predictions
                  </h3>
                  <p className="chart-subtitle">
                    Latest match predictions
                  </p>
                </div>
                <button
                  onClick={() => navigate('/predictions')}
                  className="text-blue-600 hover:text-blue-700 dark:text-blue-400"
                >
                  View All
                </button>
              </div>
              
              <div className="space-y-3 max-h-80 overflow-y-auto dashboard-scroll">
                {recentPredictions.slice(0, 6).map((prediction) => (
                  <div key={prediction.id} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-medium text-gray-900 dark:text-white">
                        {prediction.home_team} vs {prediction.away_team}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {new Date(prediction.match_date).toLocaleDateString()}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div className="text-center">
                        <div className="text-gray-600 dark:text-gray-400">Home</div>
                        <div className="font-semibold text-green-600">
                          {(prediction.home_win_prob * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-600 dark:text-gray-400">Draw</div>
                        <div className="font-semibold text-yellow-600">
                          {(prediction.draw_prob * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-600 dark:text-gray-400">Away</div>
                        <div className="font-semibold text-red-600">
                          {(prediction.away_win_prob * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-2 flex justify-between items-center">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {prediction.league_name}
                      </span>
                      <div className={`confidence-indicator confidence-indicator--${
                        Math.max(prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob) >= 0.7 ? 'high' :
                        Math.max(prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob) >= 0.5 ? 'medium' : 'low'
                      }`}>
                        {(Math.max(prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob) * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar Content */}
          <div className="space-y-6">
            {/* Live Matches */}
            <div className="chart-container">
              <div className="chart-header">
                <div>
                  <h3 className="chart-title">
                    <Activity size={20} />
                    Live Matches
                  </h3>
                  <p className="chart-subtitle">
                    Currently playing
                  </p>
                </div>
                <button
                  onClick={() => navigate('/live')}
                  className="text-blue-600 hover:text-blue-700 dark:text-blue-400"
                >
                  View All
                </button>
              </div>
              
              <div className="space-y-3 max-h-64 overflow-y-auto dashboard-scroll">
                {liveMatches.length > 0 ? (
                  liveMatches.slice(0, 5).map((match) => (
                    <div key={match.id} className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <div className="live-indicator">
                          LIVE
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {match.minute}'
                        </div>
                      </div>
                      
                      <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                        {match.home_team} vs {match.away_team}
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div className="text-lg font-bold text-gray-900 dark:text-white">
                          {match.home_score} - {match.away_score}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {match.league_name}
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    <Activity size={24} className="mx-auto mb-2 opacity-50" />
                    <p>No live matches currently</p>
                  </div>
                )}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="chart-container">
              <div className="chart-header">
                <div>
                  <h3 className="chart-title">
                    <Zap size={20} />
                    Quick Actions
                  </h3>
                </div>
              </div>
              
              <div className="space-y-2">
                <button
                  onClick={() => navigate('/analytics/comparison')}
                  className="w-full p-3 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center gap-3"
                >
                  <BarChart3 size={16} className="text-blue-600" />
                  <span>Compare Teams</span>
                </button>
                
                <button
                  onClick={() => navigate('/analytics/statistics')}
                  className="w-full p-3 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center gap-3"
                >
                  <TrendingUp size={16} className="text-green-600" />
                  <span>View Statistics</span>
                </button>
                
                <button
                  onClick={() => navigate('/predictions?confidence=high')}
                  className="w-full p-3 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center gap-3"
                >
                  <Star size={16} className="text-yellow-600" />
                  <span>Top Predictions</span>
                </button>
                
                <button
                  onClick={() => navigate('/leagues')}
                  className="w-full p-3 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors flex items-center gap-3"
                >
                  <Globe size={16} className="text-purple-600" />
                  <span>Browse Leagues</span>
                </button>
              </div>
            </div>

            {/* Model Performance Summary */}
            {modelPerformance && (
              <div className="chart-container">
                <div className="chart-header">
                  <div>
                    <h3 className="chart-title">
                      <Award size={20} />
                      Model Performance
                    </h3>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Overall Accuracy</span>
                    <span className="font-semibold text-green-600">
                      {(modelPerformance.overall_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Home Win Accuracy</span>
                    <span className="font-semibold text-blue-600">
                      {(modelPerformance.home_win_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Draw Accuracy</span>
                    <span className="font-semibold text-yellow-600">
                      {(modelPerformance.draw_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Away Win Accuracy</span>
                    <span className="font-semibold text-red-600">
                      {(modelPerformance.away_win_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="confidence-bar">
                    <div 
                      className="confidence-bar-fill confidence-bar-fill--high"
                      style={{ width: `${modelPerformance.overall_accuracy * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;