import {
    Activity,
    AlertTriangle,
    ArrowRight,
    Award,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Building,
    Calendar,
    Download,
    Flag,
    Globe,
    Heart,
    RefreshCw,
    Shield,
    Target,
    Trophy,
    Users
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    Area,
    AreaChart,
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

const TeamDetail = () => {
  const { teamId } = useParams();
  const navigate = useNavigate();
  
  // State Management
  const [team, setTeam] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [players, setPlayers] = useState([]);
  const [matches, setMatches] = useState([]);
  const [injuries, setInjuries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [favorites, setFavorites] = useState(new Set());
  
  // Performance data
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [currentSeason, setCurrentSeason] = useState('2024-25');
  const [seasonOptions, setSeasonOptions] = useState([]);
  const [standings, setStandings] = useState(null);

  // Fetch data functions
  const fetchTeam = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}`);
      if (!response.ok) throw new Error('Failed to fetch team');
      
      const data = await response.json();
      setTeam(data.team);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/statistics?season=${currentSeason}`);
      if (response.ok) {
        const data = await response.json();
        setStatistics(data.statistics);
        setPerformanceHistory(data.performance_history || []);
      }
    } catch (err) {
      console.error('Error fetching statistics:', err);
    }
  };

  const fetchPlayers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/players?season=${currentSeason}`);
      if (response.ok) {
        const data = await response.json();
        setPlayers(data.players || []);
      }
    } catch (err) {
      console.error('Error fetching players:', err);
    }
  };

  const fetchMatches = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/matches?season=${currentSeason}&limit=10`);
      if (response.ok) {
        const data = await response.json();
        setMatches(data.matches || []);
      }
    } catch (err) {
      console.error('Error fetching matches:', err);
    }
  };

  const fetchInjuries = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/injuries`);
      if (response.ok) {
        const data = await response.json();
        setInjuries(data.injuries || []);
      }
    } catch (err) {
      console.error('Error fetching injuries:', err);
    }
  };

  const fetchStandings = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/standings?season=${currentSeason}`);
      if (response.ok) {
        const data = await response.json();
        setStandings(data.standings);
      }
    } catch (err) {
      console.error('Error fetching standings:', err);
    }
  };

  const fetchSeasons = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/teams/${teamId}/seasons`);
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
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/teams`);
      if (response.ok) {
        const data = await response.json();
        setFavorites(new Set(data.teams?.map(t => t.id) || []));
      }
    } catch (err) {
      console.error('Error fetching favorites:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    if (teamId) {
      fetchTeam();
      fetchSeasons();
      fetchFavorites();
    }
  }, [teamId]);

  // Fetch season-dependent data
  useEffect(() => {
    if (teamId && currentSeason) {
      fetchStatistics();
      fetchPlayers();
      fetchMatches();
      fetchInjuries();
      fetchStandings();
    }
  }, [teamId, currentSeason]);

  // Handle favorite toggle
  const handleFavoriteToggle = async () => {
    try {
      const isFavorite = favorites.has(parseInt(teamId));
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/teams/${teamId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(parseInt(teamId));
        } else {
          newFavorites.add(parseInt(teamId));
        }
        setFavorites(newFavorites);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Navigation handlers
  const handlePlayerClick = (playerId) => {
    navigate(`/players/${playerId}`);
  };

  const handleLeagueClick = (leagueId) => {
    navigate(`/leagues/${leagueId}`);
  };

  const handleMatchClick = (matchId) => {
    navigate(`/matches/${matchId}`);
  };

  const handleVenueClick = () => {
    if (team?.venue) {
      // Could navigate to venue details or show on map
      console.log('Navigate to venue:', team.venue);
    }
  };

  // Export functionality
  const handleExport = () => {
    const dataToExport = {
      team,
      statistics,
      players,
      matches,
      injuries,
      standings,
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
    link.download = `${team?.name?.replace(/\s+/g, '_')}_${currentSeason}_analysis.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Calculate team age
  const calculateAverageAge = () => {
    if (!players.length) return 0;
    const totalAge = players.reduce((sum, player) => {
      const age = calculateAge(player.date_of_birth);
      return sum + (age || 0);
    }, 0);
    return totalAge / players.length;
  };

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

  // Get form indicator
  const getFormColor = (form) => {
    if (!form) return 'gray';
    const wins = form.split('').filter(r => r === 'W').length;
    const total = form.length;
    const winRate = wins / total;
    
    if (winRate >= 0.8) return 'green';
    if (winRate >= 0.6) return 'blue';
    if (winRate >= 0.4) return 'yellow';
    return 'red';
  };

  // Statistics cards
  const statsCards = statistics ? [
    {
      id: 'goals_scored',
      icon: Target,
      label: 'Goals Scored',
      value: statistics.goals_scored || 0,
      description: `${currentSeason} season`,
      change: statistics.goals_scored_change || 0,
      changeType: (statistics.goals_scored_change || 0) >= 0 ? 'positive' : 'negative'
    },
    {
      id: 'goals_conceded',
      icon: Shield,
      label: 'Goals Conceded',
      value: statistics.goals_conceded || 0,
      description: `${currentSeason} season`,
      change: statistics.goals_conceded_change || 0,
      changeType: (statistics.goals_conceded_change || 0) <= 0 ? 'positive' : 'negative'
    },
    {
      id: 'wins',
      icon: Trophy,
      label: 'Wins',
      value: statistics.wins || 0,
      description: `out of ${statistics.matches_played || 0} matches`,
      progress: ((statistics.wins || 0) / (statistics.matches_played || 1)) * 100,
      progressColor: 'green'
    },
    {
      id: 'league_position',
      icon: Award,
      label: 'League Position',
      value: standings?.position ? `${standings.position}` : 'N/A',
      description: standings?.league_name || 'Current season',
      change: standings?.position_change || 0,
      changeType: (standings?.position_change || 0) <= 0 ? 'positive' : 'negative'
    }
  ] : [];

  // Team performance radar data
  const radarData = statistics ? [
    { attribute: 'Attack', value: (statistics.goals_scored || 0) / (statistics.matches_played || 1) * 20 },
    { attribute: 'Defense', value: Math.max(0, 100 - ((statistics.goals_conceded || 0) / (statistics.matches_played || 1) * 25)) },
    { attribute: 'Possession', value: (statistics.possession_avg || 0) },
    { attribute: 'Pass Accuracy', value: (statistics.pass_accuracy_avg || 0) },
    { attribute: 'Discipline', value: Math.max(0, 100 - ((statistics.yellow_cards || 0) + (statistics.red_cards || 0) * 2)) },
    { attribute: 'Form', value: (statistics.win_rate || 0) }
  ] : [];

  // Active injuries
  const activeInjuries = injuries.filter(injury => injury.status === 'active');

  if (loading) {
    return (
      <div className="p-6">
        <Loading context="teams" message="Loading team profile..." />
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
              Error loading team
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchTeam}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!team) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <Users className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Team not found
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            The requested team could not be found
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
                onClick={() => navigate('/teams')}
                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 flex items-center space-x-1"
              >
                <ArrowRight className="w-4 h-4 rotate-180" />
                <span>Back to Teams</span>
              </button>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              {team.name}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Complete team analysis and player squad
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
              {favorites.has(parseInt(teamId)) ? (
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
                fetchTeam();
                fetchStatistics();
                fetchPlayers();
                fetchMatches();
                fetchInjuries();
                fetchStandings();
              }}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Team Profile Card */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex flex-col lg:flex-row lg:items-start space-y-6 lg:space-y-0 lg:space-x-8">
            {/* Team Logo and Basic Info */}
            <div className="flex-shrink-0 text-center lg:text-left">
              <img
                src={team.logo || '/api/placeholder/120/120'}
                alt={team.name}
                className="w-32 h-32 mx-auto lg:mx-0 rounded-lg"
              />
              
              {/* Current Form */}
              {statistics?.form && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Recent Form
                  </h4>
                  <div className="flex justify-center lg:justify-start space-x-1">
                    {statistics.form.split('').map((result, index) => (
                      <div
                        key={index}
                        className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                          result === 'W' ? 'bg-green-500' :
                          result === 'D' ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                      >
                        {result}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Injury Alert */}
              {activeInjuries.length > 0 && (
                <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                    <span className="text-sm font-medium text-red-800 dark:text-red-300">
                      {activeInjuries.length} Injured Player{activeInjuries.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Team Details */}
            <div className="flex-1">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Basic Information */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Basic Information
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Flag className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Country:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{team.country}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Founded:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {team.founded || 'Unknown'}
                      </span>
                    </div>
                    
                    {team.venue && (
                      <button
                        onClick={handleVenueClick}
                        className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                      >
                        <Building className="w-4 h-4" />
                        <span className="text-sm font-medium">{team.venue}</span>
                      </button>
                    )}
                    
                    {team.capacity && (
                      <div className="flex items-center space-x-2">
                        <Users className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">Capacity:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {team.capacity.toLocaleString()}
                        </span>
                      </div>
                    )}
                    
                    <div className="flex items-center space-x-2">
                      <Globe className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Website:</span>
                      {team.website ? (
                        <a
                          href={team.website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                        >
                          Official Site
                        </a>
                      ) : (
                        <span className="font-medium text-gray-900 dark:text-white">N/A</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Current League */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Current Competition
                  </h3>
                  <button
                    onClick={() => handleLeagueClick(team.league?.id)}
                    className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-3 rounded-lg transition-colors w-full text-left"
                  >
                    <img
                      src={team.league?.logo || '/api/placeholder/32/32'}
                      alt={team.league?.name}
                      className="w-8 h-8 rounded"
                    />
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {team.league?.name}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {team.league?.country}
                      </div>
                    </div>
                  </button>
                  
                  {standings && (
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Current Position:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {standings.position}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Points:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {standings.points}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Goal Difference:</span>
                        <span className={`font-medium ${
                          (standings.goal_difference || 0) >= 0 
                            ? 'text-green-600 dark:text-green-400' 
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {standings.goal_difference > 0 ? '+' : ''}{standings.goal_difference}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Squad Information */}
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Squad Information
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Users className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Squad Size:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {players.length} players
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Average Age:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {calculateAverageAge().toFixed(1)} years
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Heart className="w-4 h-4 text-red-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Injured:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {activeInjuries.length} player{activeInjuries.length !== 1 ? 's' : ''}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Flag className="w-4 h-4 text-purple-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Internationals:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {players.filter(p => p.international_caps > 0).length} player{players.filter(p => p.international_caps > 0).length !== 1 ? 's' : ''}
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
            if (statId === 'league_position') {
              setActiveTab('standings');
            }
          }}
        />

        {/* Tabs Navigation */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Performance Overview', icon: BarChart3 },
                { id: 'squad', label: 'Squad & Players', icon: Users },
                { id: 'matches', label: 'Recent Matches', icon: Activity },
                { id: 'injuries', label: 'Injury Report', icon: Heart },
                { id: 'analysis', label: 'Team Analysis', icon: Target }
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
                    {/* Goals Trend */}
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Goals This Season
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={performanceHistory}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Area 
                            type="monotone" 
                            dataKey="goals_scored" 
                            stackId="1"
                            stroke="#10B981" 
                            fill="#10B981"
                            fillOpacity={0.3}
                            name="Goals Scored"
                          />
                          <Area 
                            type="monotone" 
                            dataKey="goals_conceded" 
                            stackId="2"
                            stroke="#EF4444" 
                            fill="#EF4444"
                            fillOpacity={0.3}
                            name="Goals Conceded"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Points Progression */}
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Points Progression
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
                            dataKey="points" 
                            stroke="#3B82F6" 
                            strokeWidth={3}
                            name="Total Points"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Team Performance Summary */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-500 rounded-full p-2">
                        <Target className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-green-600 dark:text-green-400 font-medium">Attack</div>
                        <div className="text-lg font-bold text-green-900 dark:text-green-100">
                          {((statistics?.goals_scored || 0) / (statistics?.matches_played || 1)).toFixed(1)} GPG
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-500 rounded-full p-2">
                        <Shield className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-blue-600 dark:text-blue-400 font-medium">Defense</div>
                        <div className="text-lg font-bold text-blue-900 dark:text-blue-100">
                          {((statistics?.goals_conceded || 0) / (statistics?.matches_played || 1)).toFixed(1)} GCA
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-purple-500 rounded-full p-2">
                        <Trophy className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-purple-600 dark:text-purple-400 font-medium">Win Rate</div>
                        <div className="text-lg font-bold text-purple-900 dark:text-purple-100">
                          {(((statistics?.wins || 0) / (statistics?.matches_played || 1)) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-orange-500 rounded-full p-2">
                        <BarChart3 className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-orange-600 dark:text-orange-400 font-medium">Form</div>
                        <div className="text-lg font-bold text-orange-900 dark:text-orange-100">
                          {statistics?.form || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Squad & Players Tab */}
            {activeTab === 'squad' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Current Squad ({currentSeason})
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {players.length} players
                  </span>
                </div>

                {players.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {players.map((player, index) => (
                      <div
                        key={index}
                        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => handlePlayerClick(player.id)}
                      >
                        <div className="flex items-center space-x-3">
                          <img
                            src={player.photo || '/api/placeholder/40/40'}
                            alt={player.name}
                            className="w-10 h-10 rounded-full"
                          />
                          <div className="flex-1">
                            <div className="font-medium text-gray-900 dark:text-white">
                              {player.name}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {player.position} • #{player.jersey_number || 'N/A'}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium text-gray-900 dark:text-white">
                              {calculateAge(player.date_of_birth)}y
                            </div>
                            {player.is_injured && (
                              <AlertTriangle className="w-4 h-4 text-red-500 ml-auto" />
                            )}
                          </div>
                        </div>
                        
                        {/* Player Stats */}
                        <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                          <div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {player.season_goals || 0}
                            </div>
                            <div className="text-gray-500 dark:text-gray-400">Goals</div>
                          </div>
                          <div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {player.season_assists || 0}
                            </div>
                            <div className="text-gray-500 dark:text-gray-400">Assists</div>
                          </div>
                          <div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {(player.average_rating || 0).toFixed(1)}
                            </div>
                            <div className="text-gray-500 dark:text-gray-400">Rating</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Users className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Squad Data
                    </h4>
                    <p className="text-gray-500 dark:text-gray-400">
                      No player data available for the selected season
                    </p>
                  </div>
                )}
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
                          
                          {/* Match Result */}
                          <div className="flex items-center space-x-4">
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                              match.result === 'W' ? 'bg-green-500' :
                              match.result === 'D' ? 'bg-yellow-500' : 'bg-red-500'
                            }`}>
                              {match.result}
                            </div>
                            
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {match.venue_type === 'home' ? 'Home' : 'Away'}
                            </div>
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

            {/* Injury Report Tab */}
            {activeTab === 'injuries' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    Team Injury Report
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {activeInjuries.length} active, {injuries.length} total
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
                            <button
                              onClick={() => handlePlayerClick(injury.player?.id)}
                              className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-2 rounded-lg transition-colors"
                            >
                              <img
                                src={injury.player?.photo || '/api/placeholder/32/32'}
                                alt={injury.player?.name}
                                className="w-8 h-8 rounded-full"
                              />
                              <div>
                                <div className="font-medium text-gray-900 dark:text-white">
                                  {injury.player?.name}
                                </div>
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                  {injury.player?.position}
                                </div>
                              </div>
                            </button>
                            
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
                                {injury.expected_return ? ` Return: ${new Date(injury.expected_return).toLocaleDateString()}` : ' TBD'}
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
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Heart className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Injuries Reported
                    </h4>
                    <p className="text-gray-500 dark:text-gray-400">
                      This team currently has no injury reports
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Team Analysis Tab */}
            {activeTab === 'analysis' && (
              <div className="space-y-6">
                {/* Team Performance Radar */}
                {radarData.length > 0 && (
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Team Performance Analysis
                    </h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <RadarChart data={radarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="attribute" />
                        <PolarRadiusAxis domain={[0, 100]} />
                        <Radar
                          name="Performance"
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

                {/* Detailed Statistics */}
                {statistics && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Offensive Stats */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                      <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Offensive Statistics
                      </h4>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Goals Scored:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {statistics.goals_scored || 0}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Goals per Game:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {((statistics.goals_scored || 0) / (statistics.matches_played || 1)).toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Shots per Game:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {((statistics.shots || 0) / (statistics.matches_played || 1)).toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Shot Accuracy:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(((statistics.shots_on_target || 0) / (statistics.shots || 1)) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Defensive Stats */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                      <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                        Defensive Statistics
                      </h4>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Goals Conceded:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {statistics.goals_conceded || 0}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Goals Conceded per Game:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {((statistics.goals_conceded || 0) / (statistics.matches_played || 1)).toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Clean Sheets:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {statistics.clean_sheets || 0}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">Clean Sheet %:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(((statistics.clean_sheets || 0) / (statistics.matches_played || 1)) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default TeamDetail;