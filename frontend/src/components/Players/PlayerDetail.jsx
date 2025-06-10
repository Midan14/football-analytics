import {
    Activity,
    AlertTriangle,
    ArrowRight,
    Award,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Calendar,
    Clock,
    Download,
    Flag,
    Footprints,
    Heart,
    RefreshCw,
    Star,
    Target,
    TrendingUp,
    User,
    Users
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
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

const PlayerDetail = () => {
  const { playerId } = useParams();
  const navigate = useNavigate();
  
  // State Management
  const [player, setPlayer] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [injuries, setInjuries] = useState([]);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [favorites, setFavorites] = useState(new Set());
  
  // Performance data
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [currentSeason, setCurrentSeason] = useState('2024-25');
  const [seasonOptions, setSeasonOptions] = useState([]);

  // Fetch data functions
  const fetchPlayer = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/players/${playerId}`);
      if (!response.ok) throw new Error('Failed to fetch player');
      
      const data = await response.json();
      setPlayer(data.player);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/players/${playerId}/statistics?season=${currentSeason}`);
      if (response.ok) {
        const data = await response.json();
        setStatistics(data.statistics);
        setPerformanceHistory(data.performance_history || []);
      }
    } catch (err) {
      console.error('Error fetching statistics:', err);
    }
  };

  const fetchInjuries = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/players/${playerId}/injuries`);
      if (response.ok) {
        const data = await response.json();
        setInjuries(data.injuries || []);
      }
    } catch (err) {
      console.error('Error fetching injuries:', err);
    }
  };

  const fetchMatches = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/players/${playerId}/matches?season=${currentSeason}&limit=10`);
      if (response.ok) {
        const data = await response.json();
        setMatches(data.matches || []);
      }
    } catch (err) {
      console.error('Error fetching matches:', err);
    }
  };

  const fetchSeasons = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/players/${playerId}/seasons`);
      if (response.ok) {
        const data = await response.json();
        setSeasonOptions(data.seasons || []);
      }
    } catch (err) {
      console.error('Error fetching seasons:', err);
    }
  };

  const fetchFavorites = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/players`);
      if (response.ok) {
        const data = await response.json();
        setFavorites(new Set(data.players?.map(p => p.id) || []));
      }
    } catch (err) {
      console.error('Error fetching favorites:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    if (playerId) {
      fetchPlayer();
      fetchSeasons();
      fetchFavorites();
    }
  }, [playerId]);

  // Fetch season-dependent data
  useEffect(() => {
    if (playerId && currentSeason) {
      fetchStatistics();
      fetchMatches();
      fetchInjuries();
    }
  }, [playerId, currentSeason]);

  // Handle favorite toggle
  const handleFavoriteToggle = async () => {
    try {
      const isFavorite = favorites.has(parseInt(playerId));
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/players/${playerId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(parseInt(playerId));
        } else {
          newFavorites.add(parseInt(playerId));
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
      player,
      statistics,
      injuries,
      matches,
      performance_history: performanceHistory,
      season: currentSeason,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${player?.name?.replace(/\s+/g, '_')}_${currentSeason}_stats.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Calculate age
  const calculateAge = (birthDate) => {
    if (!birthDate) return null;
    const today = new Date();
    const birth = new Date(birthDate);
    let age = today.getFullYear() - birth.getFullYear();
    const monthDiff = today.getMonth() - birth.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
      age--;
    }
    return age;
  };

  // Get position color
  const getPositionColor = (position) => {
    const colors = {
      'GK': 'yellow',
      'DEF': 'blue',
      'MID': 'green',
      'FWD': 'red'
    };
    return colors[position] || 'gray';
  };

  // Statistics cards
  const statsCards = statistics ? [
    {
      id: 'goals',
      icon: Target,
      label: 'Goals',
      value: statistics.goals || 0,
      description: `${currentSeason} season`,
      change: statistics.goals_change || 0,
      changeType: (statistics.goals_change || 0) >= 0 ? 'positive' : 'negative'
    },
    {
      id: 'assists',
      icon: Users,
      label: 'Assists',
      value: statistics.assists || 0,
      description: `${currentSeason} season`,
      change: statistics.assists_change || 0,
      changeType: (statistics.assists_change || 0) >= 0 ? 'positive' : 'negative'
    },
    {
      id: 'minutes',
      icon: Clock,
      label: 'Minutes Played',
      value: (statistics.minutes_played || 0).toLocaleString(),
      description: 'Total this season',
      progress: ((statistics.minutes_played || 0) / (statistics.possible_minutes || 1)) * 100,
      progressColor: 'blue'
    },
    {
      id: 'rating',
      icon: Star,
      label: 'Average Rating',
      value: (statistics.average_rating || 0).toFixed(1),
      description: 'Performance rating',
      progress: ((statistics.average_rating || 0) / 10) * 100,
      progressColor: statistics.average_rating >= 7 ? 'green' : statistics.average_rating >= 6 ? 'yellow' : 'red'
    }
  ] : [];

  // Radar chart data for player attributes
  const radarData = statistics ? [
    { attribute: 'Pace', value: statistics.pace || 0 },
    { attribute: 'Shooting', value: statistics.shooting || 0 },
    { attribute: 'Passing', value: statistics.passing || 0 },
    { attribute: 'Dribbling', value: statistics.dribbling || 0 },
    { attribute: 'Defending', value: statistics.defending || 0 },
    { attribute: 'Physical', value: statistics.physical || 0 }
  ] : [];

  // Active injuries
  const activeInjuries = injuries.filter(injury => injury.status === 'active');

  if (loading) {
    return (
      <div className="p-6">
        <Loading context="players" message="Loading player profile..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 dark:bg-red-900/20 dark:border-red-800">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <h3 className="text-red-800 font-medium dark:text-red-300">
              Error loading player
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchPlayer}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!player) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <User className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Player not found
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            The requested player could not be found
          </p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <button
                onClick={() => navigate('/players')}
                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 flex items-center space-x-1"
              >
                <ArrowRight className="w-4 h-4 rotate-180" />
                <span>Back to Players</span>
              </button>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              {player.name}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Complete player profile and statistics
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Season Selector */}
            <select
              value={currentSeason}
              onChange={(e) => setCurrentSeason(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              {seasonOptions.map(season => (
                <option key={season} value={season}>
                  {season}
                </option>
              ))}
            </select>
            
            <button
              onClick={handleFavoriteToggle}
              className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700 transition-colors"
            >
              {favorites.has(parseInt(playerId)) ? (
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
              onClick={() => {
                fetchPlayer();
                fetchStatistics();
                fetchInjuries();
                fetchMatches();
              }}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Player Profile Card */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex flex-col lg:flex-row lg:items-start space-y-6 lg:space-y-0 lg:space-x-8">
            {/* Player Photo and Basic Info */}
            <div className="flex-shrink-0">
              <img
                src={player.photo || '/api/placeholder/120/120'}
                alt={player.name}
                className="w-32 h-32 rounded-lg mx-auto lg:mx-0"
              />
              
              {/* Injury Alert */}
              {activeInjuries.length > 0 && (
                <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                    <span className="text-sm font-medium text-red-800 dark:text-red-300">
                      Currently Injured
                    </span>
                  </div>
                  <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                    {activeInjuries[0].injury_type} - Expected return: {activeInjuries[0].expected_return ? new Date(activeInjuries[0].expected_return).toLocaleDateString() : 'TBD'}
                  </p>
                </div>
              )}
            </div>

            {/* Player Details */}
            <div className="flex-1">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Basic Information */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Basic Information
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full bg-${getPositionColor(player.position)}-500`}></div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">Position:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{player.position}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Age:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {calculateAge(player.date_of_birth) || 'Unknown'}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Flag className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Nationality:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{player.nationality}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Height:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{player.height || 'N/A'}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Weight:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{player.weight || 'N/A'}</span>
                    </div>
                  </div>
                </div>

                {/* Current Team */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Current Team
                  </h3>
                  <button
                    onClick={() => handleTeamClick(player.team?.id)}
                    className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-3 rounded-lg transition-colors w-full text-left"
                  >
                    <img
                      src={player.team?.logo || '/api/placeholder/32/32'}
                      alt={player.team?.name}
                      className="w-8 h-8 rounded"
                    />
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {player.team?.name}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {player.team?.country}
                      </div>
                    </div>
                  </button>
                  
                  {player.league && (
                    <button
                      onClick={() => handleLeagueClick(player.league?.id)}
                      className="mt-3 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 text-sm font-medium"
                    >
                      {player.league.name}
                    </button>
                  )}
                  
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Jersey Number:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        #{player.jersey_number || 'N/A'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Market Value:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.market_value || 'N/A'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Contract Until:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.contract_until ? new Date(player.contract_until).getFullYear() : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Career Highlights */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Career Highlights
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Award className="w-4 h-4 text-yellow-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Career Goals:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.career_goals || 0}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Users className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Career Assists:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.career_assists || 0}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Footprints className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Appearances:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.career_appearances || 0}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Flag className="w-4 h-4 text-red-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Int'l Caps:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {player.international_caps || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        <StatsCards 
          stats={statsCards}
          loading={!statistics}
          error={null}
          onCardClick={(statId) => {
            // Could navigate to specific stat analysis
          }}
        />

        {/* Tabs Navigation */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Performance Overview', icon: BarChart3 },
                { id: 'statistics', label: 'Detailed Stats', icon: Target },
                { id: 'matches', label: 'Recent Matches', icon: Activity },
                { id: 'injuries', label: 'Injury History', icon: Heart },
                { id: 'attributes', label: 'Player Attributes', icon: Star }
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
            {/* Performance Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Performance Charts */}
                {performanceHistory.length > 0 && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Goals Over Time */}
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Goals This Season
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={performanceHistory}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="goals" 
                            stroke="#10B981" 
                            strokeWidth={2}
                            name="Goals"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Assists Over Time */}
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Assists This Season
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={performanceHistory}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="assists" 
                            stroke="#3B82F6" 
                            strokeWidth={2}
                            name="Assists"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Form Summary */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-500 rounded-full p-2">
                        <TrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-green-600 dark:text-green-400 font-medium">Current Form</div>
                        <div className="text-lg font-bold text-green-900 dark:text-green-100">
                          {statistics?.current_form || 'Good'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-500 rounded-full p-2">
                        <BarChart3 className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">Last 5 Games</div>
                        <div className="text-lg font-bold text-blue-900 dark:text-blue-100">
                          {statistics?.goals_last_5 || 0}G {statistics?.assists_last_5 || 0}A
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-purple-500 rounded-full p-2">
                        <Star className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-purple-600 dark:text-purple-400 font-medium">Match Rating</div>
                        <div className="text-lg font-bold text-purple-900 dark:text-purple-100">
                          {(statistics?.average_rating || 0).toFixed(1)}/10
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Detailed Statistics Tab */}
            {activeTab === 'statistics' && statistics && (
              <div className="space-y-6">
                {/* Offensive Stats */}
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Offensive Statistics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.goals || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Goals</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.assists || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Assists</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.shots || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Shots</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.shots_on_target || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">On Target</div>
                    </div>
                  </div>
                </div>

                {/* Passing & Playmaking */}
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Passing & Playmaking
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {((statistics.pass_accuracy || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Pass Accuracy</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.key_passes || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Key Passes</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.crosses || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Crosses</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.through_balls || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Through Balls</div>
                    </div>
                  </div>
                </div>

                {/* Defensive Stats */}
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Defensive Statistics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.tackles || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Tackles</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.interceptions || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Interceptions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.clearances || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Clearances</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statistics.blocks || 0}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">Blocks</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Recent Matches Tab */}
            {activeTab === 'matches' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Recent Matches ({currentSeason})
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    Last {matches.length} games
                  </span>
                </div>

                {matches.length > 0 ? (
                  <div className="space-y-4">
                    {matches.map((match, index) => (
                      <div
                        key={index}
                        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => handleMatchClick(match.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {new Date(match.date).toLocaleDateString()}
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <img
                                src={match.home_team?.logo || '/api/placeholder/20/20'}
                                alt={match.home_team?.name}
                                className="w-5 h-5 rounded"
                              />
                              <span className="font-medium">{match.home_team?.name}</span>
                            </div>
                            
                            <div className="text-lg font-bold">
                              {match.home_score} - {match.away_score}
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <span className="font-medium">{match.away_team?.name}</span>
                              <img
                                src={match.away_team?.logo || '/api/placeholder/20/20'}
                                alt={match.away_team?.name}
                                className="w-5 h-5 rounded"
                              />
                            </div>
                          </div>
                          
                          {/* Player Performance */}
                          <div className="flex items-center space-x-4 text-sm">
                            {match.player_stats?.goals > 0 && (
                              <div className="flex items-center space-x-1">
                                <Target className="w-4 h-4 text-green-500" />
                                <span>{match.player_stats.goals}</span>
                              </div>
                            )}
                            {match.player_stats?.assists > 0 && (
                              <div className="flex items-center space-x-1">
                                <Users className="w-4 h-4 text-blue-500" />
                                <span>{match.player_stats.assists}</span>
                              </div>
                            )}
                            {match.player_stats?.rating && (
                              <div className="flex items-center space-x-1">
                                <Star className="w-4 h-4 text-yellow-500" />
                                <span>{match.player_stats.rating}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Activity className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Recent Matches
                    </h4>
                    <p className="text-gray-500 dark:text-gray-400">
                      No match data available for the selected season
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Injury History Tab */}
            {activeTab === 'injuries' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Injury History
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {injuries.length} total injuries
                  </span>
                </div>

                {injuries.length > 0 ? (
                  <div className="space-y-4">
                    {injuries.map((injury, index) => (
                      <div
                        key={index}
                        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className={`p-2 rounded-lg ${
                              injury.status === 'active' ? 'bg-red-100 dark:bg-red-900/20' : 'bg-green-100 dark:bg-green-900/20'
                            }`}>
                              <AlertTriangle className={`w-5 h-5 ${
                                injury.status === 'active' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
                              }`} />
                            </div>
                            
                            <div>
                              <div className="font-medium text-gray-900 dark:text-white">
                                {injury.injury_type}
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                {new Date(injury.date).toLocaleDateString()} - 
                                {injury.expected_return ? ` Expected return: ${new Date(injury.expected_return).toLocaleDateString()}` : ' Ongoing'}
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                              injury.status === 'active' 
                                ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300'
                                : 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300'
                            }`}>
                              {injury.status}
                            </span>
                            
                            <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                              injury.severity === 'minor' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300' :
                              injury.severity === 'moderate' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-300' :
                              'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300'
                            }`}>
                              {injury.severity}
                            </span>
                          </div>
                        </div>
                        
                        {injury.description && (
                          <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                            <p className="text-sm text-gray-700 dark:text-gray-300">
                              {injury.description}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Heart className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Injury History
                    </h4>
                    <p className="text-gray-500 dark:text-gray-400">
                      This player has no recorded injuries
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Player Attributes Tab */}
            {activeTab === 'attributes' && (
              <div className="space-y-6">
                {/* Radar Chart */}
                {radarData.length > 0 && (
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Player Attributes Radar
                    </h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <RadarChart data={radarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="attribute" />
                        <PolarRadiusAxis domain={[0, 100]} />
                        <Radar
                          name="Attributes"
                          dataKey="value"
                          stroke="#8B5CF6"
                          fill="#8B5CF6"
                          fillOpacity={0.3}
                        />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Attribute Breakdown */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {radarData.map((attr, index) => (
                    <div key={index} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                          {attr.attribute}
                        </h4>
                        <span className="text-2xl font-bold text-gray-900 dark:text-white">
                          {attr.value}
                        </span>
                      </div>
                      
                      <div className="w-full bg-gray-200 rounded-full h-3 dark:bg-gray-600">
                        <div
                          className={`h-3 rounded-full transition-all duration-300 ${
                            attr.value >= 80 ? 'bg-green-500' : 
                            attr.value >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${attr.value}%` }}
                        ></div>
                      </div>
                      
                      <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                        {attr.value >= 80 ? 'Excellent' : 
                         attr.value >= 60 ? 'Good' : 
                         attr.value >= 40 ? 'Average' : 'Needs Improvement'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default PlayerDetail;