import {
    Activity,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Calendar,
    ChevronDown,
    ChevronUp,
    Clock,
    Download,
    Eye,
    Filter,
    Globe,
    Grid,
    List,
    MapPin,
    Play,
    RefreshCw,
    Search,
    SortAsc,
    SortDesc,
    Target,
    Trophy
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const MatchList = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State Management
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Filter States
  const [filters, setFilters] = useState({
    status: 'all', // all, live, upcoming, finished
    league: '',
    confederation: '',
    country: '',
    date: '',
    timeframe: 'today' // today, tomorrow, week, month
  });
  
  // UI States
  const [viewMode, setViewMode] = useState('cards'); // cards, table, compact
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);
  const [showFilters, setShowFilters] = useState(false);
  const [expandedMatch, setExpandedMatch] = useState(null);
  
  // Data for filters
  const [leagues, setLeagues] = useState([]);
  const [countries, setCountries] = useState([]);
  const [favorites, setFavorites] = useState(new Set());
  
  // Statistics
  const [statistics, setStatistics] = useState({
    total_matches: 0,
    live_matches: 0,
    today_matches: 0,
    prediction_accuracy: 0
  });

  // Fetch data functions
  const fetchMatches = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams({
        q: searchQuery,
        status: filters.status,
        league: filters.league,
        confederation: filters.confederation,
        country: filters.country,
        date: filters.date,
        timeframe: filters.timeframe,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: currentPage,
        per_page: itemsPerPage
      });

      const response = await fetch(`${API_BASE_URL}/api/matches?${queryParams}`);
      if (!response.ok) throw new Error('Failed to fetch matches');
      
      const data = await response.json();
      setMatches(data.matches || []);
      setStatistics(data.statistics || {});
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchFilterData = async () => {
    try {
      const [leaguesRes, countriesRes, favoritesRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/leagues`),
        fetch(`${API_BASE_URL}/api/countries`),
        fetch(`${API_BASE_URL}/api/user/favorites/matches`)
      ]);

      if (leaguesRes.ok) {
        const leaguesData = await leaguesRes.json();
        setLeagues(leaguesData.leagues || []);
      }

      if (countriesRes.ok) {
        const countriesData = await countriesRes.json();
        setCountries(countriesData.countries || []);
      }

      if (favoritesRes.ok) {
        const favoritesData = await favoritesRes.json();
        setFavorites(new Set(favoritesData.matches?.map(m => m.id) || []));
      }
    } catch (err) {
      console.error('Error fetching filter data:', err);
    }
  };

  // Auto-refresh for live matches
  useEffect(() => {
    let interval;
    if (autoRefresh && filters.status === 'live') {
      interval = setInterval(fetchMatches, 30000); // 30 seconds
    }
    return () => clearInterval(interval);
  }, [autoRefresh, filters.status]);

  // Initial data fetch
  useEffect(() => {
    fetchMatches();
    fetchFilterData();
  }, [searchQuery, filters, sortBy, sortOrder, currentPage]);

  // Handle favorite toggle
  const handleFavoriteToggle = async (matchId) => {
    try {
      const isFavorite = favorites.has(matchId);
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/matches/${matchId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(matchId);
        } else {
          newFavorites.add(matchId);
        }
        setFavorites(newFavorites);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Handle navigation
  const handleMatchClick = (matchId) => {
    navigate(`/matches/${matchId}`);
  };

  const handleLiveMatchClick = (matchId) => {
    navigate(`/live/${matchId}`);
  };

  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
  };

  const handleLeagueClick = (leagueId) => {
    navigate(`/leagues/${leagueId}`);
  };

  // Export functionality
  const handleExport = () => {
    const dataToExport = {
      matches: filteredMatches,
      filters,
      statistics,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `matches_${filters.timeframe}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Filter and sort matches
  const filteredMatches = useMemo(() => {
    let filtered = matches.filter(match => {
      const matchesSearch = !searchQuery || 
        match.home_team?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        match.away_team?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        match.league?.name?.toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesSearch;
    });

    // Sort matches
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      if (sortBy === 'date') {
        aValue = new Date(a.date);
        bValue = new Date(b.date);
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [matches, searchQuery, sortBy, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredMatches.length / itemsPerPage);
  const paginatedMatches = filteredMatches.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Statistics cards
  const statsCards = [
    {
      id: 'total_matches',
      type: 'matches',
      icon: Calendar,
      label: 'Total Matches',
      value: statistics.total_matches || 0,
      description: 'In selected timeframe',
      change: 5.2,
      changeType: 'positive'
    },
    {
      id: 'live_matches',
      type: 'live',
      icon: Activity,
      label: 'Live Matches',
      value: statistics.live_matches || 0,
      description: 'Currently playing',
      isLive: true,
      actions: [
        { 
          label: 'View All', 
          icon: Eye, 
          onClick: () => setFilters(prev => ({ ...prev, status: 'live' }))
        }
      ]
    },
    {
      id: 'today_matches',
      type: 'today',
      icon: Clock,
      label: "Today's Matches",
      value: statistics.today_matches || 0,
      description: 'Scheduled for today',
      change: 2.1,
      changeType: 'positive'
    },
    {
      id: 'prediction_accuracy',
      type: 'predictions',
      icon: Target,
      label: 'Prediction Accuracy',
      value: `${(statistics.prediction_accuracy || 0)}%`,
      description: 'AI model performance',
      progress: statistics.prediction_accuracy || 0,
      progressColor: 'green'
    }
  ];

  // Confederations for filtering
  const confederations = [
    { value: 'UEFA', label: 'UEFA (Europe)', color: 'blue' },
    { value: 'CONMEBOL', label: 'CONMEBOL (South America)', color: 'green' },
    { value: 'CONCACAF', label: 'CONCACAF (North America)', color: 'yellow' },
    { value: 'AFC', label: 'AFC (Asia)', color: 'red' },
    { value: 'CAF', label: 'CAF (Africa)', color: 'orange' },
    { value: 'OFC', label: 'OFC (Oceania)', color: 'purple' }
  ];

  // Status options
  const statusOptions = [
    { value: 'all', label: 'All Matches', icon: Globe },
    { value: 'live', label: 'Live', icon: Activity },
    { value: 'upcoming', label: 'Upcoming', icon: Clock },
    { value: 'finished', label: 'Finished', icon: Trophy }
  ];

  // Timeframe options
  const timeframeOptions = [
    { value: 'today', label: 'Today' },
    { value: 'tomorrow', label: 'Tomorrow' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' }
  ];

  // Get match status styling
  const getMatchStatusStyle = (status) => {
    switch (status) {
      case 'live':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300';
      case 'upcoming':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300';
      case 'finished':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300';
      case 'halftime':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-300';
    }
  };

  // Format time
  const formatMatchTime = (date, status) => {
    const matchDate = new Date(date);
    const now = new Date();
    
    if (status === 'live') {
      return 'LIVE';
    }
    
    if (status === 'finished') {
      return 'FT';
    }

    // If today, show time only
    if (matchDate.toDateString() === now.toDateString()) {
      return matchDate.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    }
    
    // If this year, show month and day
    if (matchDate.getFullYear() === now.getFullYear()) {
      return matchDate.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric'
      });
    }
    
    // Otherwise show full date
    return matchDate.toLocaleDateString('en-US');
  };

  if (loading && matches.length === 0) {
    return (
      <div className="p-6">
        <Loading context="matches" message="Loading matches..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 dark:bg-red-900/20 dark:border-red-800">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 text-red-500">⚠️</div>
            <h3 className="text-red-800 font-medium dark:text-red-300">
              Error loading matches
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchMatches}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
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
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              Football Matches
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Live matches, upcoming fixtures, and results from {leagues.length}+ leagues worldwide
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Auto-refresh toggle */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="auto-refresh"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label htmlFor="auto-refresh" className="text-sm text-gray-700 dark:text-gray-300">
                Auto-refresh
              </label>
            </div>
            
            {/* Last updated */}
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
            
            {/* Refresh button */}
            <button
              onClick={fetchMatches}
              disabled={loading}
              className="p-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Statistics Cards */}
        <StatsCards 
          stats={statsCards}
          loading={loading}
          error={null}
          onCardClick={(statId) => {
            if (statId === 'live_matches') {
              setFilters(prev => ({ ...prev, status: 'live' }));
            }
          }}
        />

        {/* Search and Filters */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          {/* Search Bar */}
          <div className="flex flex-col lg:flex-row lg:items-center space-y-4 lg:space-y-0 lg:space-x-4 mb-4">
            <div className="flex-1 relative">
              <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search teams, leagues, or matches..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
            </div>
            
            <div className="flex items-center space-x-2">
              {/* View Mode Toggle */}
              <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('cards')}
                  className={`p-2 rounded ${viewMode === 'cards' 
                    ? 'bg-white dark:bg-gray-600 shadow' 
                    : 'text-gray-600 dark:text-gray-400'}`}
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('table')}
                  className={`p-2 rounded ${viewMode === 'table' 
                    ? 'bg-white dark:bg-gray-600 shadow' 
                    : 'text-gray-600 dark:text-gray-400'}`}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
              
              {/* Filters Toggle */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Filter className="w-4 h-4" />
                <span>Filters</span>
                {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              
              {/* Export Button */}
              <button
                onClick={handleExport}
                className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>
            </div>
          </div>
          
          {/* Filters Panel */}
          {showFilters && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Status Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Status
                  </label>
                  <select
                    value={filters.status}
                    onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {statusOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Timeframe Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Timeframe
                  </label>
                  <select
                    value={filters.timeframe}
                    onChange={(e) => setFilters(prev => ({ ...prev, timeframe: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {timeframeOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Confederation Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Confederation
                  </label>
                  <select
                    value={filters.confederation}
                    onChange={(e) => setFilters(prev => ({ ...prev, confederation: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    <option value="">All Confederations</option>
                    {confederations.map(conf => (
                      <option key={conf.value} value={conf.value}>
                        {conf.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* League Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    League
                  </label>
                  <select
                    value={filters.league}
                    onChange={(e) => setFilters(prev => ({ ...prev, league: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    <option value="">All Leagues</option>
                    {leagues.map(league => (
                      <option key={league.id} value={league.id}>
                        {league.name} ({league.country})
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              
              {/* Clear Filters */}
              <div className="flex justify-end">
                <button
                  onClick={() => setFilters({
                    status: 'all',
                    league: '',
                    confederation: '',
                    country: '',
                    date: '',
                    timeframe: 'today'
                  })}
                  className="text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                >
                  Clear all filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Info */}
        <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
          <div>
            Showing {paginatedMatches.length} of {filteredMatches.length} matches
          </div>
          
          {/* Sort Controls */}
          <div className="flex items-center space-x-4">
            <label className="font-medium">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="date">Date</option>
              <option value="league">League</option>
              <option value="status">Status</option>
            </select>
            
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Matches Display */}
        {viewMode === 'cards' ? (
          /* Cards View */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {paginatedMatches.map(match => (
              <div
                key={match.id}
                className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => match.status === 'live' ? handleLiveMatchClick(match.id) : handleMatchClick(match.id)}
              >
                {/* Match Header */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span
                        className={`px-2 py-1 text-xs font-medium border rounded-full ${getMatchStatusStyle(match.status)}`}
                      >
                        {formatMatchTime(match.date, match.status)}
                        {match.status === 'live' && (
                          <span className="ml-1 inline-block w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"></span>
                        )}
                      </span>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleLeagueClick(match.league?.id);
                        }}
                        className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 font-medium"
                      >
                        {match.league?.name}
                      </button>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleFavoriteToggle(match.id);
                      }}
                      className="text-gray-400 hover:text-yellow-500 transition-colors"
                    >
                      {favorites.has(match.id) ? (
                        <BookmarkCheck className="w-5 h-5 text-yellow-500" />
                      ) : (
                        <Bookmark className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Teams and Score */}
                <div className="p-6">
                  <div className="flex items-center justify-between">
                    {/* Home Team */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleTeamClick(match.home_team?.id);
                      }}
                      className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-2 rounded-lg transition-colors flex-1"
                    >
                      <img
                        src={match.home_team?.logo || '/api/placeholder/32/32'}
                        alt={match.home_team?.name}
                        className="w-8 h-8 rounded"
                      />
                      <span className="font-medium text-gray-900 dark:text-white text-left">
                        {match.home_team?.name}
                      </span>
                    </button>

                    {/* Score */}
                    <div className="flex items-center space-x-4 px-4">
                      {match.status === 'upcoming' ? (
                        <div className="text-2xl font-bold text-gray-400">
                          VS
                        </div>
                      ) : (
                        <div className="text-center">
                          <div className="text-2xl font-bold text-gray-900 dark:text-white">
                            {match.home_score} - {match.away_score}
                          </div>
                          {match.status === 'live' && match.minute && (
                            <div className="text-xs text-red-600 font-medium">
                              {match.minute}'
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Away Team */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleTeamClick(match.away_team?.id);
                      }}
                      className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-2 rounded-lg transition-colors flex-1 justify-end"
                    >
                      <span className="font-medium text-gray-900 dark:text-white text-right">
                        {match.away_team?.name}
                      </span>
                      <img
                        src={match.away_team?.logo || '/api/placeholder/32/32'}
                        alt={match.away_team?.name}
                        className="w-8 h-8 rounded"
                      />
                    </button>
                  </div>

                  {/* Match Info */}
                  {match.venue && (
                    <div className="mt-4 flex items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                      <MapPin className="w-4 h-4 mr-1" />
                      {match.venue}
                    </div>
                  )}

                  {/* AI Prediction */}
                  {match.prediction && (
                    <div className="mt-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-3">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-600 dark:text-gray-400">AI Prediction</span>
                        <span className="text-purple-600 dark:text-purple-400 font-medium">
                          {(match.prediction.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                      <div className="mt-2 flex items-center space-x-2">
                        <div className="flex-1 text-center">
                          <div className="text-xs text-gray-500">Home</div>
                          <div className="font-medium">{(match.prediction.home_win * 100).toFixed(1)}%</div>
                        </div>
                        <div className="flex-1 text-center">
                          <div className="text-xs text-gray-500">Draw</div>
                          <div className="font-medium">{(match.prediction.draw * 100).toFixed(1)}%</div>
                        </div>
                        <div className="flex-1 text-center">
                          <div className="text-xs text-gray-500">Away</div>
                          <div className="font-medium">{(match.prediction.away_win * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Quick Actions */}
                <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700/50 rounded-b-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                      {match.status === 'live' && (
                        <span className="flex items-center">
                          <Activity className="w-3 h-3 mr-1" />
                          Live
                        </span>
                      )}
                      {match.prediction && (
                        <span className="flex items-center">
                          <Target className="w-3 h-3 mr-1" />
                          AI Prediction
                        </span>
                      )}
                      <span className="flex items-center">
                        <BarChart3 className="w-3 h-3 mr-1" />
                        Stats
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {match.status === 'live' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleLiveMatchClick(match.id);
                          }}
                          className="flex items-center space-x-1 px-2 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors"
                        >
                          <Play className="w-3 h-3" />
                          <span>Watch Live</span>
                        </button>
                      )}
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleMatchClick(match.id);
                        }}
                        className="flex items-center space-x-1 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Details</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Table View */
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Time/Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Match
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      League
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      AI Prediction
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {paginatedMatches.map(match => (
                    <tr
                      key={match.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                      onClick={() => match.status === 'live' ? handleLiveMatchClick(match.id) : handleMatchClick(match.id)}
                    >
                      {/* Time/Status */}
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <span className={`px-2 py-1 text-xs font-medium border rounded-full ${getMatchStatusStyle(match.status)}`}>
                            {formatMatchTime(match.date, match.status)}
                            {match.status === 'live' && (
                              <span className="ml-1 inline-block w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"></span>
                            )}
                          </span>
                        </div>
                      </td>

                      {/* Match */}
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-4">
                          {/* Home Team */}
                          <div className="flex items-center space-x-2 flex-1">
                            <img
                              src={match.home_team?.logo || '/api/placeholder/24/24'}
                              alt={match.home_team?.name}
                              className="w-6 h-6 rounded"
                            />
                            <span className="font-medium text-gray-900 dark:text-white truncate">
                              {match.home_team?.name}
                            </span>
                          </div>
                          
                          <span className="text-gray-400 text-sm">vs</span>
                          
                          {/* Away Team */}
                          <div className="flex items-center space-x-2 flex-1 justify-end">
                            <span className="font-medium text-gray-900 dark:text-white truncate">
                              {match.away_team?.name}
                            </span>
                            <img
                              src={match.away_team?.logo || '/api/placeholder/24/24'}
                              alt={match.away_team?.name}
                              className="w-6 h-6 rounded"
                            />
                          </div>
                        </div>
                      </td>

                      {/* Score */}
                      <td className="px-6 py-4 text-center">
                        {match.status === 'upcoming' ? (
                          <span className="text-gray-400">-</span>
                        ) : (
                          <div>
                            <div className="text-lg font-bold text-gray-900 dark:text-white">
                              {match.home_score} - {match.away_score}
                            </div>
                            {match.status === 'live' && match.minute && (
                              <div className="text-xs text-red-600 font-medium">
                                {match.minute}'
                              </div>
                            )}
                          </div>
                        )}
                      </td>

                      {/* League */}
                      <td className="px-6 py-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleLeagueClick(match.league?.id);
                          }}
                          className="text-blue-600 hover:text-blue-800 dark:text-blue-400 font-medium"
                        >
                          {match.league?.name}
                        </button>
                      </td>

                      {/* AI Prediction */}
                      <td className="px-6 py-4 text-center">
                        {match.prediction ? (
                          <div className="text-sm">
                            <div className="text-purple-600 dark:text-purple-400 font-medium">
                              {match.prediction.home_win > match.prediction.away_win && match.prediction.home_win > match.prediction.draw ? 'Home' :
                               match.prediction.away_win > match.prediction.draw ? 'Away' : 'Draw'}
                            </div>
                            <div className="text-xs text-gray-500">
                              {(Math.max(match.prediction.home_win, match.prediction.away_win, match.prediction.draw) * 100).toFixed(1)}%
                            </div>
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleFavoriteToggle(match.id);
                            }}
                            className="text-gray-400 hover:text-yellow-500 transition-colors"
                          >
                            {favorites.has(match.id) ? (
                              <BookmarkCheck className="w-4 h-4 text-yellow-500" />
                            ) : (
                              <Bookmark className="w-4 h-4" />
                            )}
                          </button>
                          
                          {match.status === 'live' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleLiveMatchClick(match.id);
                              }}
                              className="p-1 text-red-600 hover:text-red-800 transition-colors"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleMatchClick(match.id);
                            }}
                            className="p-1 text-blue-600 hover:text-blue-800 transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty State */}
        {filteredMatches.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <Calendar className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No matches found
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              Try adjusting your search criteria or filters
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters({
                  status: 'all',
                  league: '',
                  confederation: '',
                  country: '',
                  date: '',
                  timeframe: 'today'
                });
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Clear all filters
            </button>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Page {currentPage} of {totalPages}
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
                className="px-3 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700"
              >
                Previous
              </button>
              
              {/* Page Numbers */}
              <div className="flex items-center space-x-1">
                {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                  const pageNum = currentPage <= 3 ? i + 1 : currentPage - 2 + i;
                  if (pageNum > totalPages) return null;
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-2 rounded-lg ${
                        currentPage === pageNum
                          ? 'bg-blue-600 text-white'
                          : 'border border-gray-300 hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700'
                      }`}
                    >
                      {pageNum}
                    </button>
                  );
                })}
              </div>
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default MatchList;