import {
    Activity,
    AlertCircle,
    ArrowLeft,
    BarChart3,
    Calendar,
    ChevronDown,
    ChevronUp,
    Download,
    Eye, Grid, List,
    Minus,
    RefreshCw,
    Shield,
    Star,
    Target,
    TrendingDown,
    TrendingUp,
    Trophy,
    Users
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    Area,
    AreaChart,
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
    XAxis, YAxis
} from 'recharts';

// Reusable Components
import ErrorBoundary from '../common/ErrorBoundary';
import Loading from '../common/Loading';
import StatsCards from '../common/StatsCards';

const TeamStats = () => {
  const { teamId } = useParams();
  const navigate = useNavigate();
  
  // State Management
  const [teamStats, setTeamStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedSeason, setSelectedSeason] = useState('2024');
  const [viewMode, setViewMode] = useState('cards');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isFavorite, setIsFavorite] = useState(false);
  const [comparisons, setComparisons] = useState([]);
  const [filters, setFilters] = useState({
    competition: 'all',
    venue: 'all', // home/away/all
    timeframe: 'season', // season/last_10/last_5
    statType: 'all'
  });

  // Available seasons for selector
  const [availableSeasons] = useState(['2024', '2023', '2022', '2021', '2020']);

  // Fetch team statistics data
  const fetchTeamStats = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const queryParams = new URLSearchParams({
        season: selectedSeason,
        competition: filters.competition,
        venue: filters.venue,
        timeframe: filters.timeframe
      });

      const [statsResponse, favoritesResponse] = await Promise.all([
        fetch(`/api/teams/${teamId}/statistics?${queryParams}`),
        fetch('/api/user/favorites/teams')
      ]);

      if (!statsResponse.ok) {
        throw new Error(`Failed to fetch team stats: ${statsResponse.status}`);
      }

      const statsData = await statsResponse.json();
      const favoritesData = favoritesResponse.ok ? await favoritesResponse.json() : [];

      setTeamStats(statsData);
      setIsFavorite(favoritesData.some(fav => fav.team_id === teamId));
      setLastUpdated(new Date());

    } catch (err) {
      console.error('Error fetching team stats:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [teamId, selectedSeason, filters]);

  // Auto-refresh effect
  useEffect(() => {
    fetchTeamStats();
  }, [fetchTeamStats]);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchTeamStats, 60000); // 1 minute
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, fetchTeamStats]);

  // Toggle favorite team
  const handleFavoriteToggle = async () => {
    try {
      const method = isFavorite ? 'DELETE' : 'POST';
      const response = await fetch(`/api/user/favorites/teams/${teamId}`, {
        method,
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        setIsFavorite(!isFavorite);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Export statistics data
  const handleExport = () => {
    if (!teamStats) return;

    const exportData = {
      team_info: teamStats.team_info,
      season: selectedSeason,
      filters,
      statistics: teamStats,
      exported_at: new Date().toISOString(),
      exported_by: 'Football Analytics Platform'
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${teamStats.team_info.name}_stats_${selectedSeason}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Navigation handlers
  const handleTeamClick = () => {
    navigate(`/teams/${teamId}`);
  };

  const handleLeagueClick = () => {
    if (teamStats?.team_info?.league_id) {
      navigate(`/leagues/${teamStats.team_info.league_id}`);
    }
  };

  const handleCompareTeam = (compareTeamId) => {
    navigate(`/teams/compare/${teamId}/${compareTeamId}`);
  };

  // Chart colors
  const chartColors = {
    primary: '#3b82f6',
    secondary: '#10b981',
    accent: '#f59e0b',
    danger: '#ef4444',
    success: '#22c55e',
    warning: '#eab308'
  };

  // Performance data for charts
  const getPerformanceData = () => {
    if (!teamStats?.performance_timeline) return [];
    
    return teamStats.performance_timeline.map((item, index) => ({
      match: `Match ${index + 1}`,
      goals_scored: item.goals_scored,
      goals_conceded: item.goals_conceded,
      points: item.result === 'W' ? 3 : item.result === 'D' ? 1 : 0,
      possession: item.possession_percentage,
      shots: item.shots_total,
      date: item.date
    }));
  };

  // Radar chart data for team attributes
  const getRadarData = () => {
    if (!teamStats?.team_attributes) return [];

    const attributes = teamStats.team_attributes;
    return [
      {
        attribute: 'Attack',
        value: attributes.attack_rating || 0,
        fullMark: 100
      },
      {
        attribute: 'Defense',
        value: attributes.defense_rating || 0,
        fullMark: 100
      },
      {
        attribute: 'Midfield',
        value: attributes.midfield_rating || 0,
        fullMark: 100
      },
      {
        attribute: 'Possession',
        value: attributes.possession_rating || 0,
        fullMark: 100
      },
      {
        attribute: 'Discipline',
        value: attributes.discipline_rating || 0,
        fullMark: 100
      },
      {
        attribute: 'Form',
        value: attributes.form_rating || 0,
        fullMark: 100
      }
    ];
  };

  // Goals breakdown data
  const getGoalsBreakdown = () => {
    if (!teamStats?.goals_breakdown) return [];

    const breakdown = teamStats.goals_breakdown;
    return [
      { name: 'Home Goals', value: breakdown.home_goals || 0, color: chartColors.primary },
      { name: 'Away Goals', value: breakdown.away_goals || 0, color: chartColors.secondary },
      { name: 'Penalties', value: breakdown.penalty_goals || 0, color: chartColors.accent },
      { name: 'Set Pieces', value: breakdown.set_piece_goals || 0, color: chartColors.warning }
    ];
  };

  // Calculate trend indicators
  const getTrendIndicator = (current, previous) => {
    if (!previous) return { icon: Minus, color: 'text-gray-400', text: 'No change' };
    
    if (current > previous) {
      return { icon: TrendingUp, color: 'text-green-500', text: 'Improving' };
    } else if (current < previous) {
      return { icon: TrendingDown, color: 'text-red-500', text: 'Declining' };
    }
    return { icon: Minus, color: 'text-gray-400', text: 'Stable' };
  };

  if (loading) {
    return <Loading context="team statistics" />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4">
        <div className="max-w-7xl mx-auto">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <div className="flex items-center space-x-2 text-red-600 dark:text-red-400">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">Error loading team statistics</span>
            </div>
            <p className="mt-2 text-red-600 dark:text-red-400">{error}</p>
            <button
              onClick={fetchTeamStats}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!teamStats) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4">
        <div className="max-w-7xl mx-auto">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
            <div className="flex items-center space-x-2 text-yellow-600 dark:text-yellow-400">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">No statistics found</span>
            </div>
            <p className="mt-2 text-yellow-600 dark:text-yellow-400">
              Statistics for this team are not available for the selected season.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const stats = teamStats.season_stats || {};
  const teamInfo = teamStats.team_info || {};

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header Section */}
        <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="max-w-7xl mx-auto px-4 py-6">
            {/* Navigation and Actions */}
            <div className="flex items-center justify-between mb-6">
              <button
                onClick={() => navigate('/teams')}
                className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Teams</span>
              </button>

              <div className="flex items-center space-x-3">
                {/* Season Selector */}
                <select
                  value={selectedSeason}
                  onChange={(e) => setSelectedSeason(e.target.value)}
                  className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {availableSeasons.map(season => (
                    <option key={season} value={season}>{season} Season</option>
                  ))}
                </select>

                {/* View Mode Toggle */}
                <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('cards')}
                    className={`p-2 rounded ${viewMode === 'cards' ? 'bg-white dark:bg-gray-600 shadow' : 'hover:bg-gray-200 dark:hover:bg-gray-600'} transition-colors`}
                  >
                    <Grid className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('table')}
                    className={`p-2 rounded ${viewMode === 'table' ? 'bg-white dark:bg-gray-600 shadow' : 'hover:bg-gray-200 dark:hover:bg-gray-600'} transition-colors`}
                  >
                    <List className="h-4 w-4" />
                  </button>
                </div>

                {/* Action Buttons */}
                <button
                  onClick={handleFavoriteToggle}
                  className={`p-2 rounded-lg transition-colors ${isFavorite ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'}`}
                >
                  <Star className={`h-5 w-5 ${isFavorite ? 'fill-current' : ''}`} />
                </button>

                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg transition-colors ${autoRefresh ? 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'}`}
                >
                  <RefreshCw className={`h-5 w-5 ${autoRefresh ? 'animate-spin' : ''}`} />
                </button>

                <button
                  onClick={handleExport}
                  className="p-2 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  <Download className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Team Information Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <img
                  src={teamInfo.logo_url || '/images/default-team.png'}
                  alt={teamInfo.name}
                  className="w-16 h-16 object-contain"
                  onError={(e) => {
                    e.target.src = '/images/default-team.png';
                  }}
                />
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                    {teamInfo.name} Statistics
                  </h1>
                  <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600 dark:text-gray-400">
                    <button
                      onClick={handleLeagueClick}
                      className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                    >
                      {teamInfo.league_name}
                    </button>
                    <span>•</span>
                    <span>{teamInfo.country}</span>
                    <span>•</span>
                    <span>{selectedSeason} Season</span>
                  </div>
                </div>
              </div>

              <div className="text-right">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                  League Position
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  #{stats.league_position || 'N/A'}
                </div>
                {stats.position_change && (
                  <div className={`text-sm flex items-center justify-end space-x-1 ${stats.position_change > 0 ? 'text-green-600' : stats.position_change < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                    {stats.position_change > 0 ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : stats.position_change < 0 ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <Minus className="h-4 w-4" />
                    )}
                    <span>{Math.abs(stats.position_change)}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Last Updated */}
            {lastUpdated && (
              <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
                Last updated: {lastUpdated.toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="max-w-7xl mx-auto px-4 py-6">
          <StatsCards
            stats={[
              {
                title: 'Goals Scored',
                value: stats.goals_scored || 0,
                change: stats.goals_scored_change,
                icon: Target,
                color: 'green'
              },
              {
                title: 'Goals Conceded',
                value: stats.goals_conceded || 0,
                change: stats.goals_conceded_change,
                icon: Shield,
                color: 'red'
              },
              {
                title: 'Win Rate',
                value: `${((stats.wins / (stats.matches_played || 1)) * 100).toFixed(1)}%`,
                change: stats.win_rate_change,
                icon: Trophy,
                color: 'blue'
              },
              {
                title: 'Points',
                value: stats.points || 0,
                change: stats.points_change,
                icon: BarChart3,
                color: 'purple'
              }
            ]}
          />

          {/* Tabs Navigation */}
          <div className="mt-8 border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8">
              {[
                { id: 'overview', label: 'Performance Overview', icon: BarChart3 },
                { id: 'attacking', label: 'Attacking Stats', icon: Target },
                { id: 'defensive', label: 'Defensive Stats', icon: Shield },
                { id: 'analysis', label: 'Team Analysis', icon: Activity },
                { id: 'comparisons', label: 'Comparisons', icon: Users }
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="mt-8">
            {activeTab === 'overview' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Goals Timeline */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Goals Timeline
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={getPerformanceData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="match" />
                      <YAxis />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="goals_scored"
                        stackId="1"
                        stroke={chartColors.success}
                        fill={chartColors.success}
                        fillOpacity={0.6}
                        name="Goals Scored"
                      />
                      <Area
                        type="monotone"
                        dataKey="goals_conceded"
                        stackId="2"
                        stroke={chartColors.danger}
                        fill={chartColors.danger}
                        fillOpacity={0.6}
                        name="Goals Conceded"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Points Progression */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Points Progression
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={getPerformanceData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="match" />
                      <YAxis />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="points"
                        stroke={chartColors.primary}
                        strokeWidth={3}
                        dot={{ fill: chartColors.primary, strokeWidth: 2, r: 4 }}
                        name="Points Gained"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Goals Breakdown */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Goals Breakdown
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={getGoalsBreakdown()}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {getGoalsBreakdown().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Team Attributes Radar */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Team Attributes
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={getRadarData()}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="attribute" />
                      <PolarRadiusAxis 
                        angle={60} 
                        domain={[0, 100]} 
                        tick={false}
                      />
                      <Radar
                        name="Rating"
                        dataKey="value"
                        stroke={chartColors.primary}
                        fill={chartColors.primary}
                        fillOpacity={0.3}
                        strokeWidth={2}
                      />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {activeTab === 'attacking' && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {/* Attacking Statistics Cards */}
                {[
                  { label: 'Goals Scored', value: stats.goals_scored || 0, trend: getTrendIndicator(stats.goals_scored, stats.previous_goals_scored) },
                  { label: 'Assists', value: stats.assists || 0, trend: getTrendIndicator(stats.assists, stats.previous_assists) },
                  { label: 'Shots on Target', value: stats.shots_on_target || 0, trend: getTrendIndicator(stats.shots_on_target, stats.previous_shots_on_target) },
                  { label: 'Shots Total', value: stats.shots_total || 0, trend: getTrendIndicator(stats.shots_total, stats.previous_shots_total) },
                  { label: 'Shot Accuracy', value: `${((stats.shots_on_target / (stats.shots_total || 1)) * 100).toFixed(1)}%`, trend: getTrendIndicator(stats.shots_on_target / (stats.shots_total || 1), stats.previous_shot_accuracy) },
                  { label: 'Goals per Game', value: (stats.goals_scored / (stats.matches_played || 1)).toFixed(2), trend: getTrendIndicator(stats.goals_scored / (stats.matches_played || 1), stats.previous_goals_per_game) }
                ].map((stat, index) => {
                  const TrendIcon = stat.trend.icon;
                  return (
                    <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {stat.label}
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                            {stat.value}
                          </p>
                        </div>
                        <div className={`flex items-center space-x-1 ${stat.trend.color}`}>
                          <TrendIcon className="h-4 w-4" />
                          <span className="text-xs">{stat.trend.text}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {activeTab === 'defensive' && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {/* Defensive Statistics Cards */}
                {[
                  { label: 'Goals Conceded', value: stats.goals_conceded || 0, trend: getTrendIndicator(-(stats.goals_conceded || 0), -(stats.previous_goals_conceded || 0)) },
                  { label: 'Clean Sheets', value: stats.clean_sheets || 0, trend: getTrendIndicator(stats.clean_sheets, stats.previous_clean_sheets) },
                  { label: 'Tackles', value: stats.tackles || 0, trend: getTrendIndicator(stats.tackles, stats.previous_tackles) },
                  { label: 'Interceptions', value: stats.interceptions || 0, trend: getTrendIndicator(stats.interceptions, stats.previous_interceptions) },
                  { label: 'Blocks', value: stats.blocks || 0, trend: getTrendIndicator(stats.blocks, stats.previous_blocks) },
                  { label: 'Goals Conceded per Game', value: (stats.goals_conceded / (stats.matches_played || 1)).toFixed(2), trend: getTrendIndicator(-(stats.goals_conceded / (stats.matches_played || 1)), -(stats.previous_goals_conceded_per_game || 0)) }
                ].map((stat, index) => {
                  const TrendIcon = stat.trend.icon;
                  return (
                    <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {stat.label}
                          </p>
                          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                            {stat.value}
                          </p>
                        </div>
                        <div className={`flex items-center space-x-1 ${stat.trend.color}`}>
                          <TrendIcon className="h-4 w-4" />
                          <span className="text-xs">{stat.trend.text}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {activeTab === 'analysis' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Performance Analysis */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
                    Performance Analysis
                  </h3>
                  <div className="space-y-4">
                    {[
                      { label: 'Attack Efficiency', value: ((stats.goals_scored / (stats.shots_total || 1)) * 100).toFixed(1), max: 100, color: 'bg-green-500' },
                      { label: 'Defensive Solidity', value: (100 - ((stats.goals_conceded / (stats.matches_played || 1)) * 10)).toFixed(1), max: 100, color: 'bg-blue-500' },
                      { label: 'Possession Control', value: stats.average_possession || 0, max: 100, color: 'bg-purple-500' },
                      { label: 'Pass Accuracy', value: stats.pass_accuracy || 0, max: 100, color: 'bg-yellow-500' },
                      { label: 'Discipline', value: (100 - ((stats.yellow_cards + stats.red_cards * 2) / (stats.matches_played || 1) * 5)).toFixed(1), max: 100, color: 'bg-indigo-500' }
                    ].map((metric, index) => (
                      <div key={index}>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {metric.label}
                          </span>
                          <span className="text-sm font-bold text-gray-900 dark:text-white">
                            {metric.value}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${metric.color}`}
                            style={{ width: `${Math.min(metric.value, metric.max)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Season Summary */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
                    Season Summary
                  </h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {stats.wins || 0}
                      </div>
                      <div className="text-sm text-green-600 dark:text-green-400">Wins</div>
                    </div>
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                      <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                        {stats.draws || 0}
                      </div>
                      <div className="text-sm text-yellow-600 dark:text-yellow-400">Draws</div>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                      <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                        {stats.losses || 0}
                      </div>
                      <div className="text-sm text-red-600 dark:text-red-400">Losses</div>
                    </div>
                  </div>
                  
                  <div className="mt-6 space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Goal Difference</span>
                      <span className={`font-medium ${(stats.goals_scored - stats.goals_conceded) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {stats.goals_scored - stats.goals_conceded > 0 ? '+' : ''}{stats.goals_scored - stats.goals_conceded}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Points per Game</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {(stats.points / (stats.matches_played || 1)).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Win Rate</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {((stats.wins / (stats.matches_played || 1)) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'comparisons' && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Team Comparisons
                  </h3>
                  <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    Add Comparison
                  </button>
                </div>
                
                {comparisons.length === 0 ? (
                  <div className="text-center py-12">
                    <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No Comparisons Yet
                    </h4>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Add teams to compare their statistics side by side.
                    </p>
                    <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                      Compare Teams
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Comparison content would go here */}
                    <p className="text-gray-600 dark:text-gray-400">
                      Team comparison functionality coming soon...
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Quick Actions */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={handleTeamClick}
              className="flex items-center justify-center space-x-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <Eye className="h-5 w-5 text-gray-600 dark:text-gray-400" />
              <span className="font-medium text-gray-900 dark:text-white">View Team Profile</span>
            </button>
            
            <button
              onClick={handleLeagueClick}
              className="flex items-center justify-center space-x-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <Trophy className="h-5 w-5 text-gray-600 dark:text-gray-400" />
              <span className="font-medium text-gray-900 dark:text-white">View League</span>
            </button>
            
            <button
              onClick={() => navigate(`/teams/${teamId}/matches`)}
              className="flex items-center justify-center space-x-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <Calendar className="h-5 w-5 text-gray-600 dark:text-gray-400" />
              <span className="font-medium text-gray-900 dark:text-white">View Matches</span>
            </button>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default TeamStats;