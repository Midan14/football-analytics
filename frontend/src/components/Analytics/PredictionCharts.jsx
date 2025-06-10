import {
    Activity,
    AlertCircle,
    Brain,
    Calendar,
    CheckCircle,
    Download,
    Percent,
    RefreshCw,
    Target,
    TrendingUp,
    Trophy,
    Zap
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const PredictionCharts = () => {
  // Estados principales
  const [predictions, setPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [selectedChart, setSelectedChart] = useState('accuracy'); // accuracy, probabilities, trends, xg
  const [dateRange, setDateRange] = useState('week'); // week, month, season
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    league_id: '',
    confederation: '',
    country: '',
    match_status: 'upcoming', // upcoming, finished, all
    confidence_min: 0.5,
    date_from: '',
    date_to: ''
  });

  // Estados para datos agregados
  const [modelPerformance, setModelPerformance] = useState(null);
  const [leagueStats, setLeagueStats] = useState([]);
  const [confidenceDistribution, setConfidenceDistribution] = useState([]);
  const [predictionTrends, setPredictionTrends] = useState([]);

  // Opciones de filtro
  const [leagues, setLeagues] = useState([]);
  const [confederations] = useState(['UEFA', 'CONMEBOL', 'CONCACAF', 'AFC', 'CAF', 'OFC']);
  
  // Configuración de gráficos
  const chartTypes = [
    { key: 'accuracy', label: 'Model Accuracy', icon: Target },
    { key: 'probabilities', label: 'Win Probabilities', icon: Percent },
    { key: 'trends', label: 'Prediction Trends', icon: TrendingUp },
    { key: 'xg', label: 'xG Predictions', icon: Activity },
    { key: 'confidence', label: 'Confidence Distribution', icon: Brain },
    { key: 'leagues', label: 'League Performance', icon: Trophy }
  ];

  const colors = {
    home_win: '#10B981',   // Green
    draw: '#F59E0B',       // Yellow
    away_win: '#EF4444',   // Red
    xg: '#3B82F6',         // Blue
    confidence: '#8B5CF6',  // Purple
    accuracy: '#06B6D4'     // Cyan
  };

  // API Functions
  const fetchPredictions = async (filters) => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams({
        ...filters,
        include_statistics: true,
        include_model_data: true
      }).toString();
      
      const response = await fetch(`${API_BASE_URL}/predictions?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.predictions || [];
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError('Error loading predictions. Please try again.');
      return [];
    } finally {
      setLoading(false);
    }
  };

  const fetchModelPerformance = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/model-performance`);
      if (response.ok) {
        const data = await response.json();
        setModelPerformance(data.performance);
      }
    } catch (error) {
      console.error('Error fetching model performance:', error);
    }
  };

  const fetchLeagueStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/league-stats`);
      if (response.ok) {
        const data = await response.json();
        setLeagueStats(data.league_stats || []);
      }
    } catch (error) {
      console.error('Error fetching league stats:', error);
    }
  };

  const fetchPredictionTrends = async () => {
    try {
      const queryParams = new URLSearchParams({
        period: dateRange,
        ...filters
      }).toString();
      
      const response = await fetch(`${API_BASE_URL}/predictions/trends?${queryParams}`);
      if (response.ok) {
        const data = await response.json();
        setPredictionTrends(data.trends || []);
      }
    } catch (error) {
      console.error('Error fetching prediction trends:', error);
    }
  };

  // Fetch leagues for filter
  useEffect(() => {
    const fetchLeagues = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/leagues?is_active=true`);
        if (response.ok) {
          const data = await response.json();
          setLeagues(data.leagues || []);
        }
      } catch (error) {
        console.error('Error fetching leagues:', error);
      }
    };

    fetchLeagues();
    fetchModelPerformance();
    fetchLeagueStats();
  }, []);

  // Fetch predictions when filters change
  useEffect(() => {
    const loadPredictions = async () => {
      const data = await fetchPredictions(filters);
      setPredictions(data);
    };

    loadPredictions();
    fetchPredictionTrends();
  }, [filters, dateRange]);

  // Filter predictions based on current filters
  useEffect(() => {
    let filtered = [...predictions];

    // Apply confidence filter
    if (filters.confidence_min > 0) {
      filtered = filtered.filter(p => 
        Math.max(p.home_win_prob, p.draw_prob, p.away_win_prob) >= filters.confidence_min
      );
    }

    // Apply date range filter
    if (filters.date_from) {
      filtered = filtered.filter(p => new Date(p.match_date) >= new Date(filters.date_from));
    }
    if (filters.date_to) {
      filtered = filtered.filter(p => new Date(p.match_date) <= new Date(filters.date_to));
    }

    setFilteredPredictions(filtered);

    // Generate confidence distribution
    const distribution = generateConfidenceDistribution(filtered);
    setConfidenceDistribution(distribution);
  }, [predictions, filters]);

  // Generate confidence distribution data
  const generateConfidenceDistribution = (predictions) => {
    const ranges = [
      { label: '50-60%', min: 0.5, max: 0.6, count: 0 },
      { label: '60-70%', min: 0.6, max: 0.7, count: 0 },
      { label: '70-80%', min: 0.7, max: 0.8, count: 0 },
      { label: '80-90%', min: 0.8, max: 0.9, count: 0 },
      { label: '90-100%', min: 0.9, max: 1.0, count: 0 }
    ];

    predictions.forEach(prediction => {
      const maxProb = Math.max(
        prediction.home_win_prob, 
        prediction.draw_prob, 
        prediction.away_win_prob
      );
      
      const range = ranges.find(r => maxProb >= r.min && maxProb < r.max);
      if (range) range.count++;
    });

    return ranges;
  };

  // Prepare data for accuracy chart
  const accuracyData = useMemo(() => {
    if (!modelPerformance) return [];
    
    return [
      { metric: 'Overall Accuracy', value: modelPerformance.overall_accuracy * 100 },
      { metric: 'Home Win Accuracy', value: modelPerformance.home_win_accuracy * 100 },
      { metric: 'Draw Accuracy', value: modelPerformance.draw_accuracy * 100 },
      { metric: 'Away Win Accuracy', value: modelPerformance.away_win_accuracy * 100 },
      { metric: 'xG Accuracy', value: modelPerformance.xg_accuracy * 100 }
    ];
  }, [modelPerformance]);

  // Prepare data for probability distribution
  const probabilityData = useMemo(() => {
    if (filteredPredictions.length === 0) return [];
    
    return filteredPredictions.slice(0, 20).map(prediction => ({
      match: `${prediction.home_team_name} vs ${prediction.away_team_name}`,
      home_win: (prediction.home_win_prob * 100).toFixed(1),
      draw: (prediction.draw_prob * 100).toFixed(1),
      away_win: (prediction.away_win_prob * 100).toFixed(1),
      match_date: prediction.match_date
    }));
  }, [filteredPredictions]);

  // Prepare xG prediction data
  const xgData = useMemo(() => {
    if (filteredPredictions.length === 0) return [];
    
    return filteredPredictions.slice(0, 15).map(prediction => ({
      match: `${prediction.home_team_name.substring(0, 8)} vs ${prediction.away_team_name.substring(0, 8)}`,
      home_xg: prediction.predicted_home_xg || 0,
      away_xg: prediction.predicted_away_xg || 0,
      total_xg: (prediction.predicted_home_xg || 0) + (prediction.predicted_away_xg || 0),
      match_date: prediction.match_date
    }));
  }, [filteredPredictions]);

  // Export functionality
  const exportData = async () => {
    try {
      const exportData = {
        predictions: filteredPredictions,
        model_performance: modelPerformance,
        league_stats: leagueStats,
        confidence_distribution: confidenceDistribution,
        filters: filters,
        generated_at: new Date().toISOString()
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `football-predictions-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting data:', error);
      setError('Error exporting data. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-4xl font-bold text-gray-900 flex items-center gap-3">
              <Brain className="text-blue-600" />
              AI Prediction Analytics
            </h1>
            <div className="flex gap-3">
              <button
                onClick={exportData}
                disabled={filteredPredictions.length === 0}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-gray-700 transition-colors"
              >
                <Download size={16} />
                Export Data
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <RefreshCw size={16} />
                Refresh
              </button>
            </div>
          </div>
          <p className="text-gray-600 text-lg">
            Real-time ML prediction analytics from {leagues.length}+ leagues worldwide
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6 flex items-center gap-2">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {/* Model Performance Summary */}
        {modelPerformance && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Overall Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">
                    {(modelPerformance.overall_accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <Target className="text-green-600" size={24} />
              </div>
              <div className="mt-2">
                <div className="flex items-center gap-1 text-sm text-gray-500">
                  <TrendingUp size={12} />
                  +2.3% vs last month
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Predictions Made</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {modelPerformance.total_predictions?.toLocaleString() || 0}
                  </p>
                </div>
                <Zap className="text-blue-600" size={24} />
              </div>
              <div className="mt-2">
                <div className="flex items-center gap-1 text-sm text-gray-500">
                  <Calendar size={12} />
                  This season
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Avg Confidence</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {(modelPerformance.avg_confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <Brain className="text-purple-600" size={24} />
              </div>
              <div className="mt-2">
                <div className="flex items-center gap-1 text-sm text-gray-500">
                  <CheckCircle size={12} />
                  High confidence
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Active Leagues</p>
                  <p className="text-2xl font-bold text-orange-600">
                    {leagues.length}
                  </p>
                </div>
                <Trophy className="text-orange-600" size={24} />
              </div>
              <div className="mt-2">
                <div className="flex items-center gap-1 text-sm text-gray-500">
                  <Activity size={12} />
                  Worldwide coverage
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-6 gap-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                League
              </label>
              <select
                value={filters.league_id}
                onChange={(e) => setFilters({...filters, league_id: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">All Leagues</option>
                {leagues.map(league => (
                  <option key={league.id} value={league.id}>
                    {league.name} ({league.country})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Confederation
              </label>
              <select
                value={filters.confederation}
                onChange={(e) => setFilters({...filters, confederation: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">All Confederations</option>
                {confederations.map(conf => (
                  <option key={conf} value={conf}>{conf}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Match Status
              </label>
              <select
                value={filters.match_status}
                onChange={(e) => setFilters({...filters, match_status: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="upcoming">Upcoming Matches</option>
                <option value="finished">Finished Matches</option>
                <option value="all">All Matches</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Min Confidence
              </label>
              <select
                value={filters.confidence_min}
                onChange={(e) => setFilters({...filters, confidence_min: parseFloat(e.target.value)})}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={0}>All Predictions</option>
                <option value={0.5}>50%+ Confidence</option>
                <option value={0.6}>60%+ Confidence</option>
                <option value={0.7}>70%+ Confidence</option>
                <option value={0.8}>80%+ Confidence</option>
                <option value={0.9}>90%+ Confidence</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Date Range
              </label>
              <select
                value={dateRange}
                onChange={(e) => setDateRange(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="week">This Week</option>
                <option value="month">This Month</option>
                <option value="season">Full Season</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Chart Type
              </label>
              <select
                value={selectedChart}
                onChange={(e) => setSelectedChart(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {chartTypes.map(chart => (
                  <option key={chart.key} value={chart.key}>{chart.label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Chart Navigation */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex flex-wrap gap-3">
            {chartTypes.map(chart => {
              const IconComponent = chart.icon;
              return (
                <button
                  key={chart.key}
                  onClick={() => setSelectedChart(chart.key)}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                    selectedChart === chart.key
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <IconComponent size={16} />
                  {chart.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Chart */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-6">
            {chartTypes.find(c => c.key === selectedChart)?.label}
          </h2>
          
          {loading ? (
            <div className="flex items-center justify-center h-96">
              <RefreshCw className="animate-spin text-blue-500" size={32} />
              <span className="ml-3 text-gray-600">Loading predictions...</span>
            </div>
          ) : (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                {selectedChart === 'accuracy' && (
                  <BarChart data={accuracyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    <Bar dataKey="value" fill={colors.accuracy} />
                  </BarChart>
                )}

                {selectedChart === 'probabilities' && (
                  <BarChart data={probabilityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="match" angle={-45} textAnchor="end" height={80} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="home_win" name="Home Win %" fill={colors.home_win} />
                    <Bar dataKey="draw" name="Draw %" fill={colors.draw} />
                    <Bar dataKey="away_win" name="Away Win %" fill={colors.away_win} />
                  </BarChart>
                )}

                {selectedChart === 'trends' && (
                  <LineChart data={predictionTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke={colors.accuracy} strokeWidth={2} name="Accuracy %" />
                    <Line type="monotone" dataKey="predictions_count" stroke={colors.confidence} strokeWidth={2} name="Predictions" />
                  </LineChart>
                )}

                {selectedChart === 'xg' && (
                  <BarChart data={xgData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="match" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="home_xg" name="Home xG" fill={colors.home_win} />
                    <Bar dataKey="away_xg" name="Away xG" fill={colors.away_win} />
                  </BarChart>
                )}

                {selectedChart === 'confidence' && (
                  <BarChart data={confidenceDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill={colors.confidence} />
                  </BarChart>
                )}

                {selectedChart === 'leagues' && (
                  <BarChart data={leagueStats}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="league_name" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill={colors.accuracy} />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Statistics Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Recent Predictions</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {filteredPredictions.slice(0, 10).map((prediction, index) => (
                <div key={prediction.id} className="p-3 border border-gray-200 rounded-lg">
                  <div className="flex justify-between items-start mb-2">
                    <div className="font-medium text-gray-900">
                      {prediction.home_team_name} vs {prediction.away_team_name}
                    </div>
                    <div className="text-sm text-gray-500">
                      {new Date(prediction.match_date).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="text-center">
                      <div className="text-gray-600">Home</div>
                      <div className="font-semibold text-green-600">
                        {(prediction.home_win_prob * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-600">Draw</div>
                      <div className="font-semibold text-yellow-600">
                        {(prediction.draw_prob * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-600">Away</div>
                      <div className="font-semibold text-red-600">
                        {(prediction.away_win_prob * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Top Performing Leagues</h3>
            <div className="space-y-3">
              {leagueStats.slice(0, 8).map((league, index) => (
                <div key={league.league_id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">{league.league_name}</div>
                    <div className="text-sm text-gray-500">{league.country}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold text-green-600">
                      {(league.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-500">
                      {league.predictions_count} predictions
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCharts;