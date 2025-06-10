import {
    AlertCircle,
    ArrowRight,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Brain,
    CheckCircle,
    Clock,
    Download,
    Eye,
    Info,
    MapPin,
    Minus,
    RefreshCw,
    Target,
    TrendingDown,
    TrendingUp,
    Trophy,
    Users
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    CartesianGrid,
    Cell,
    Legend,
    Line,
    LineChart,
    Pie,
    PieChart,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis,
    Radar,
    RadarChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const MatchPrediction = () => {
  const { matchId } = useParams();
  const navigate = useNavigate();
  
  // State Management
  const [prediction, setPrediction] = useState(null);
  const [match, setMatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [favorites, setFavorites] = useState(new Set());
  
  // Model performance data
  const [modelPerformance, setModelPerformance] = useState(null);
  const [historicalAccuracy, setHistoricalAccuracy] = useState([]);
  const [similarMatches, setSimilarMatches] = useState([]);
  
  // UI States
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('overall');

  // Fetch data functions
  const fetchPrediction = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/predictions/match/${matchId}`);
      if (!response.ok) throw new Error('Failed to fetch prediction');
      
      const data = await response.json();
      setPrediction(data.prediction);
      setMatch(data.match);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchModelPerformance = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/predictions/model-performance`);
      if (response.ok) {
        const data = await response.json();
        setModelPerformance(data.performance);
        setHistoricalAccuracy(data.historical_accuracy || []);
      }
    } catch (err) {
      console.error('Error fetching model performance:', err);
    }
  };

  const fetchSimilarMatches = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/predictions/similar-matches/${matchId}`);
      if (response.ok) {
        const data = await response.json();
        setSimilarMatches(data.matches || []);
      }
    } catch (err) {
      console.error('Error fetching similar matches:', err);
    }
  };

  const fetchFavorites = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/matches`);
      if (response.ok) {
        const data = await response.json();
        setFavorites(new Set(data.matches?.map(m => m.id) || []));
      }
    } catch (err) {
      console.error('Error fetching favorites:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    if (matchId) {
      fetchPrediction();
      fetchModelPerformance();
      fetchSimilarMatches();
      fetchFavorites();
    }
  }, [matchId]);

  // Handle favorite toggle
  const handleFavoriteToggle = async () => {
    try {
      const isFavorite = favorites.has(parseInt(matchId));
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/matches/${matchId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(parseInt(matchId));
        } else {
          newFavorites.add(parseInt(matchId));
        }
        setFavorites(newFavorites);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Navigation handlers
  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
  };

  const handleLeagueClick = (leagueId) => {
    navigate(`/leagues/${leagueId}`);
  };

  const handleMatchClick = (matchId) => {
    navigate(`/matches/${matchId}`);
  };

  // Export functionality
  const handleExport = () => {
    const dataToExport = {
      match,
      prediction,
      model_performance: modelPerformance,
      similar_matches: similarMatches,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `prediction_${match?.home_team?.name}_vs_${match?.away_team?.name}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Get prediction confidence level
  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return { level: 'High', color: 'green', icon: CheckCircle };
    if (confidence >= 0.6) return { level: 'Medium', color: 'yellow', icon: AlertCircle };
    return { level: 'Low', color: 'red', icon: AlertCircle };
  };

  // Get winner prediction
  const getWinnerPrediction = () => {
    if (!prediction) return null;
    
    const { home_win, draw, away_win } = prediction.probabilities;
    const maxProb = Math.max(home_win, draw, away_win);
    
    if (maxProb === home_win) return { type: 'home', team: match?.home_team, probability: home_win };
    if (maxProb === away_win) return { type: 'away', team: match?.away_team, probability: away_win };
    return { type: 'draw', team: null, probability: draw };
  };

  // Chart data preparation
  const probabilitiesData = prediction ? [
    { name: 'Home Win', value: prediction.probabilities.home_win * 100, color: '#10B981' },
    { name: 'Draw', value: prediction.probabilities.draw * 100, color: '#F59E0B' },
    { name: 'Away Win', value: prediction.probabilities.away_win * 100, color: '#EF4444' }
  ] : [];

  const confidenceData = prediction ? [
    { metric: 'Overall', confidence: prediction.confidence * 100 },
    { metric: 'Home Form', confidence: (prediction.factors?.home_form || 0) * 100 },
    { metric: 'Away Form', confidence: (prediction.factors?.away_form || 0) * 100 },
    { metric: 'Head to Head', confidence: (prediction.factors?.head_to_head || 0) * 100 },
    { metric: 'League Position', confidence: (prediction.factors?.league_position || 0) * 100 },
    { metric: 'Recent Results', confidence: (prediction.factors?.recent_results || 0) * 100 }
  ] : [];

  // Statistics cards
  const statsCards = [
    {
      id: 'confidence',
      icon: Target,
      label: 'Prediction Confidence',
      value: prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '0%',
      description: getConfidenceLevel(prediction?.confidence || 0).level,
      progress: (prediction?.confidence || 0) * 100,
      progressColor: getConfidenceLevel(prediction?.confidence || 0).color
    },
    {
      id: 'winner',
      icon: Trophy,
      label: 'Predicted Winner',
      value: (() => {
        const winner = getWinnerPrediction();
        if (!winner) return 'N/A';
        return winner.type === 'draw' ? 'Draw' : winner.team?.name || 'Unknown';
      })(),
      description: (() => {
        const winner = getWinnerPrediction();
        return winner ? `${(winner.probability * 100).toFixed(1)}% probability` : 'No prediction';
      })(),
      variant: 'gradient'
    },
    {
      id: 'model_accuracy',
      icon: Brain,
      label: 'Model Accuracy',
      value: modelPerformance ? `${(modelPerformance.overall_accuracy * 100).toFixed(1)}%` : 'N/A',
      description: 'Overall model performance',
      change: modelPerformance?.accuracy_change || 0,
      changeType: (modelPerformance?.accuracy_change || 0) >= 0 ? 'positive' : 'negative'
    },
    {
      id: 'similar_matches',
      icon: BarChart3,
      label: 'Similar Matches',
      value: similarMatches.length,
      description: 'Historical comparisons',
      actions: [
        { 
          label: 'View All', 
          icon: Eye, 
          onClick: () => setActiveTab('similar')
        }
      ]
    }
  ];

  if (loading) {
    return (
      <div className="p-6">
        <Loading context="predictions" message="Loading AI prediction..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 dark:bg-red-900/20 dark:border-red-800">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <h3 className="text-red-800 font-medium dark:text-red-300">
              Error loading prediction
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchPrediction}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!prediction || !match) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <Brain className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No prediction available
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            AI prediction is not available for this match
          </p>
        </div>
      </div>
    );
  }

  const confidenceInfo = getConfidenceLevel(prediction.confidence);
  const ConfidenceIcon = confidenceInfo.icon;
  const winner = getWinnerPrediction();

  return (
    <ErrorBoundary>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <button
                onClick={() => navigate('/predictions')}
                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 flex items-center space-x-1"
              >
                <ArrowRight className="w-4 h-4 rotate-180" />
                <span>Back to Predictions</span>
              </button>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              AI Match Prediction
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Advanced machine learning analysis for {match.home_team?.name} vs {match.away_team?.name}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={handleFavoriteToggle}
              className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700 transition-colors"
            >
              {favorites.has(parseInt(matchId)) ? (
                <BookmarkCheck className="w-4 h-4 text-yellow-500" />
              ) : (
                <Bookmark className="w-4 h-4" />
              )}
              <span>Favorite</span>
            </button>
            
            <button
              onClick={handleExport}
              className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700 transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            
            <button
              onClick={fetchPrediction}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Match Info Card */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                match.status === 'upcoming' 
                  ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300'
                  : 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-300'
              }`}>
                <Clock className="w-4 h-4" />
                <span>{new Date(match.date).toLocaleString()}</span>
              </div>
              
              <button
                onClick={() => handleLeagueClick(match.league?.id)}
                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 font-medium"
              >
                {match.league?.name}
              </button>
            </div>
            
            {match.venue && (
              <div className="flex items-center text-gray-500 dark:text-gray-400">
                <MapPin className="w-4 h-4 mr-1" />
                <span className="text-sm">{match.venue}</span>
              </div>
            )}
          </div>

          {/* Teams Display */}
          <div className="flex items-center justify-between">
            {/* Home Team */}
            <button
              onClick={() => handleTeamClick(match.home_team?.id)}
              className="flex items-center space-x-4 hover:bg-gray-50 dark:hover:bg-gray-700 p-4 rounded-lg transition-colors flex-1"
            >
              <img
                src={match.home_team?.logo || '/api/placeholder/48/48'}
                alt={match.home_team?.name}
                className="w-12 h-12 rounded"
              />
              <div className="text-left">
                <div className="font-bold text-lg text-gray-900 dark:text-white">
                  {match.home_team?.name}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Home
                </div>
              </div>
            </button>

            {/* Prediction Summary */}
            <div className="flex-1 text-center px-8">
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6">
                <div className="flex items-center justify-center space-x-2 mb-3">
                  <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                  <span className="text-lg font-bold text-gray-900 dark:text-white">AI Prediction</span>
                </div>
                
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                  {winner.type === 'draw' ? 'Draw' : winner.team?.name}
                </div>
                
                <div className="flex items-center justify-center space-x-2 mb-3">
                  <ConfidenceIcon className={`w-5 h-5 text-${confidenceInfo.color}-500`} />
                  <span className={`text-${confidenceInfo.color}-600 dark:text-${confidenceInfo.color}-400 font-medium`}>
                    {confidenceInfo.level} Confidence
                  </span>
                </div>
                
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(winner.probability * 100).toFixed(1)}%
                </div>
                
                <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Probability
                </div>
              </div>
            </div>

            {/* Away Team */}
            <button
              onClick={() => handleTeamClick(match.away_team?.id)}
              className="flex items-center space-x-4 hover:bg-gray-50 dark:hover:bg-gray-700 p-4 rounded-lg transition-colors flex-1 justify-end"
            >
              <div className="text-right">
                <div className="font-bold text-lg text-gray-900 dark:text-white">
                  {match.away_team?.name}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Away
                </div>
              </div>
              <img
                src={match.away_team?.logo || '/api/placeholder/48/48'}
                alt={match.away_team?.name}
                className="w-12 h-12 rounded"
              />
            </button>
          </div>
        </div>

        {/* Statistics Cards */}
        <StatsCards 
          stats={statsCards}
          loading={false}
          error={null}
          onCardClick={(statId) => {
            if (statId === 'similar_matches') {
              setActiveTab('similar');
            }
          }}
        />

        {/* Tabs Navigation */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Overview', icon: Eye },
                { id: 'probabilities', label: 'Probabilities', icon: BarChart3 },
                { id: 'factors', label: 'Key Factors', icon: Target },
                { id: 'confidence', label: 'Confidence Analysis', icon: Brain },
                { id: 'similar', label: 'Similar Matches', icon: Users }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Quick Summary */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-500 rounded-full p-2">
                        <Trophy className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-green-600 dark:text-green-400 font-medium">Most Likely</div>
                        <div className="text-lg font-bold text-green-900 dark:text-green-100">
                          {winner.type === 'draw' ? 'Draw' : `${winner.team?.name} Win`}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-purple-500 rounded-full p-2">
                        <Target className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-purple-600 dark:text-purple-400 font-medium">Confidence</div>
                        <div className="text-lg font-bold text-purple-900 dark:text-purple-100">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-500 rounded-full p-2">
                        <Brain className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">Model Used</div>
                        <div className="text-lg font-bold text-blue-900 dark:text-blue-100">
                          {prediction.model_version || 'v2.1'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Probability Breakdown */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 text-center">
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                      {(prediction.probabilities.home_win * 100).toFixed(1)}%
                    </div>
                    <div className="text-green-800 dark:text-green-300 font-medium">
                      {match.home_team?.name} Win
                    </div>
                  </div>

                  <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 text-center">
                    <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-2">
                      {(prediction.probabilities.draw * 100).toFixed(1)}%
                    </div>
                    <div className="text-yellow-800 dark:text-yellow-300 font-medium">
                      Draw
                    </div>
                  </div>

                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 text-center">
                    <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">
                      {(prediction.probabilities.away_win * 100).toFixed(1)}%
                    </div>
                    <div className="text-red-800 dark:text-red-300 font-medium">
                      {match.away_team?.name} Win
                    </div>
                  </div>
                </div>

                {/* Key Insights */}
                {prediction.insights && (
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-4">
                      AI Insights
                    </h3>
                    <div className="space-y-3">
                      {prediction.insights.map((insight, index) => (
                        <div key={index} className="flex items-start space-x-3">
                          <div className="bg-blue-500 rounded-full p-1 mt-1">
                            <Info className="w-3 h-3 text-white" />
                          </div>
                          <p className="text-blue-800 dark:text-blue-200">{insight}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Probabilities Tab */}
            {activeTab === 'probabilities' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Probability Pie Chart */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Match Outcome Probabilities
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={probabilitiesData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {probabilitiesData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Probability Bars */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Probability Breakdown
                    </h3>
                    <div className="space-y-4">
                      {probabilitiesData.map((item, index) => (
                        <div key={index}>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {item.name}
                            </span>
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {item.value.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-600">
                            <div
                              className="h-2.5 rounded-full transition-all duration-300"
                              style={{ 
                                width: `${item.value}%`,
                                backgroundColor: item.color
                              }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Score Predictions */}
                {prediction.score_predictions && (
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Most Likely Score Predictions
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                      {prediction.score_predictions.slice(0, 10).map((score, index) => (
                        <div key={index} className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg border">
                          <div className="text-lg font-bold text-gray-900 dark:text-white">
                            {score.home_score} - {score.away_score}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {(score.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Key Factors Tab */}
            {activeTab === 'factors' && (
              <div className="space-y-6">
                {/* Factors Radar Chart */}
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Prediction Factors Analysis
                  </h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <RadarChart data={confidenceData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" />
                      <PolarRadiusAxis domain={[0, 100]} />
                      <Radar
                        name="Confidence"
                        dataKey="confidence"
                        stroke="#8B5CF6"
                        fill="#8B5CF6"
                        fillOpacity={0.3}
                      />
                      <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>

                {/* Detailed Factors */}
                {prediction.factors && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {Object.entries(prediction.factors).map(([factor, value]) => (
                      <div key={factor} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-lg font-medium text-gray-900 dark:text-white capitalize">
                            {factor.replace('_', ' ')}
                          </h4>
                          <div className="flex items-center space-x-2">
                            {value > 0.7 ? (
                              <TrendingUp className="w-5 h-5 text-green-500" />
                            ) : value > 0.4 ? (
                              <Minus className="w-5 h-5 text-yellow-500" />
                            ) : (
                              <TrendingDown className="w-5 h-5 text-red-500" />
                            )}
                            <span className={`font-bold ${
                              value > 0.7 ? 'text-green-600' : 
                              value > 0.4 ? 'text-yellow-600' : 'text-red-600'
                            }`}>
                              {(value * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        
                        <div className="w-full bg-gray-200 rounded-full h-3 dark:bg-gray-600">
                          <div
                            className={`h-3 rounded-full transition-all duration-300 ${
                              value > 0.7 ? 'bg-green-500' : 
                              value > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${value * 100}%` }}
                          ></div>
                        </div>
                        
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                          {getFactorDescription(factor, value)}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Confidence Analysis Tab */}
            {activeTab === 'confidence' && (
              <div className="space-y-6">
                {/* Confidence Meter */}
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6">
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                      Overall Prediction Confidence
                    </h3>
                    <div className="relative w-48 h-48 mx-auto mb-6">
                      <svg className="w-48 h-48 transform -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="transparent"
                          className="text-gray-200 dark:text-gray-700"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="transparent"
                          strokeDasharray={`${(prediction.confidence * 251.2)} 251.2`}
                          className={`text-${confidenceInfo.color}-500`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-gray-900 dark:text-white">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </div>
                          <div className={`text-${confidenceInfo.color}-600 dark:text-${confidenceInfo.color}-400 font-medium`}>
                            {confidenceInfo.level}
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-${confidenceInfo.color}-100 dark:bg-${confidenceInfo.color}-900/20`}>
                      <ConfidenceIcon className={`w-5 h-5 text-${confidenceInfo.color}-600 dark:text-${confidenceInfo.color}-400`} />
                      <span className={`text-${confidenceInfo.color}-800 dark:text-${confidenceInfo.color}-300 font-medium`}>
                        {getConfidenceDescription(prediction.confidence)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Model Performance */}
                {modelPerformance && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Model Performance Metrics
                    </h3>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {(modelPerformance.overall_accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Overall Accuracy</div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {(modelPerformance.home_accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Home Wins</div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {(modelPerformance.draw_accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Draws</div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          {(modelPerformance.away_accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Away Wins</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Historical Accuracy Trend */}
                {historicalAccuracy.length > 0 && (
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Model Accuracy Trend
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={historicalAccuracy}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="accuracy" 
                          stroke="#8B5CF6" 
                          strokeWidth={2}
                          name="Accuracy %"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Similar Matches Tab */}
            {activeTab === 'similar' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Similar Historical Matches
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {similarMatches.length} matches found
                  </span>
                </div>

                {similarMatches.length > 0 ? (
                  <div className="space-y-4">
                    {similarMatches.map((similar, index) => (
                      <div
                        key={index}
                        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => handleMatchClick(similar.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className="flex items-center space-x-2">
                              <img
                                src={similar.home_team?.logo || '/api/placeholder/24/24'}
                                alt={similar.home_team?.name}
                                className="w-6 h-6 rounded"
                              />
                              <span className="font-medium">{similar.home_team?.name}</span>
                            </div>
                            
                            <div className="text-lg font-bold text-gray-900 dark:text-white">
                              {similar.home_score} - {similar.away_score}
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <span className="font-medium">{similar.away_team?.name}</span>
                              <img
                                src={similar.away_team?.logo || '/api/placeholder/24/24'}
                                alt={similar.away_team?.name}
                                className="w-6 h-6 rounded"
                              />
                            </div>
                          </div>
                          
                          <div className="text-right">
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {new Date(similar.date).toLocaleDateString()}
                            </div>
                            <div className="text-sm text-blue-600 dark:text-blue-400">
                              {similar.similarity_score ? `${(similar.similarity_score * 100).toFixed(1)}% similar` : 'Similar context'}
                            </div>
                          </div>
                        </div>
                        
                        {similar.prediction_accuracy && (
                          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600 dark:text-gray-400">
                                Prediction was: {similar.predicted_outcome}
                              </span>
                              <span className={`font-medium ${
                                similar.prediction_accuracy > 0.8 ? 'text-green-600' : 
                                similar.prediction_accuracy > 0.5 ? 'text-yellow-600' : 'text-red-600'
                              }`}>
                                {similar.prediction_accuracy > 0.8 ? 'Correct' : 
                                 similar.prediction_accuracy > 0.5 ? 'Partially Correct' : 'Incorrect'}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Users className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Similar Matches Found
                    </h4>
                    <p className="text-gray-500 dark:text-gray-400">
                      No historical matches with similar characteristics were found
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <button
              onClick={() => navigate(`/matches/${matchId}`)}
              className="flex items-center justify-center space-x-2 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Eye className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <span className="text-blue-800 dark:text-blue-300 font-medium">View Match</span>
            </button>
            
            <button
              onClick={() => handleTeamClick(match.home_team?.id)}
              className="flex items-center justify-center space-x-2 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
            >
              <Users className="w-5 h-5 text-green-600 dark:text-green-400" />
              <span className="text-green-800 dark:text-green-300 font-medium">Home Team</span>
            </button>
            
            <button
              onClick={() => handleTeamClick(match.away_team?.id)}
              className="flex items-center justify-center space-x-2 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors"
            >
              <Users className="w-5 h-5 text-orange-600 dark:text-orange-400" />
              <span className="text-orange-800 dark:text-orange-300 font-medium">Away Team</span>
            </button>
            
            <button
              onClick={() => handleLeagueClick(match.league?.id)}
              className="flex items-center justify-center space-x-2 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
            >
              <Trophy className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <span className="text-purple-800 dark:text-purple-300 font-medium">League</span>
            </button>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

// Helper functions
const getFactorDescription = (factor, value) => {
  const descriptions = {
    home_form: value > 0.7 ? "Excellent home performance recently" : 
               value > 0.4 ? "Moderate home form" : "Poor home form lately",
    away_form: value > 0.7 ? "Strong away performance" : 
               value > 0.4 ? "Average away form" : "Struggling away from home",
    head_to_head: value > 0.7 ? "Historically favors this outcome" : 
                  value > 0.4 ? "Balanced historical record" : "Historical disadvantage",
    league_position: value > 0.7 ? "Strong league position influence" : 
                     value > 0.4 ? "Moderate position impact" : "League position less influential",
    recent_results: value > 0.7 ? "Recent momentum supports prediction" : 
                    value > 0.4 ? "Mixed recent results" : "Poor recent momentum"
  };
  return descriptions[factor] || "Factor analysis not available";
};

const getConfidenceDescription = (confidence) => {
  if (confidence >= 0.8) return "Very reliable prediction with strong supporting factors";
  if (confidence >= 0.6) return "Good prediction confidence with solid analysis";
  return "Lower confidence - many variables at play";
};

export default MatchPrediction;