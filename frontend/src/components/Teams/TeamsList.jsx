import {
    Activity,
    AlertTriangle,
    Award,
    Bookmark,
    BookmarkCheck,
    Building,
    Calendar,
    ChevronDown,
    ChevronUp,
    Download,
    Eye,
    Filter,
    Flag,
    Globe,
    Grid,
    List,
    RefreshCw,
    Search,
    SortAsc,
    SortDesc,
    Trophy,
    Users
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const TeamsList = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State Management
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Filter States
  const [filters, setFilters] = useState({
    league: '',
    confederation: '',
    country: '',
    founded_min: '',
    founded_max: '',
    capacity_min: '',
    capacity_max: '',
    division_level: '' // 1, 2, 3
  });
  
  // UI States
  const [viewMode, setViewMode] = useState('cards'); // cards, table, grid
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(24);
  const [showFilters, setShowFilters] = useState(false);
  
  // Data for filters
  const [leagues, setLeagues] = useState([]);
  const [countries, setCountries] = useState([]);
  const [favorites, setFavorites] = useState(new Set());
  
  // Statistics
  const [statistics, setStatistics] = useState({
    total_teams: 0,
    active_teams: 0,
    leagues_covered: 0,
    countries_covered: 0,
    avg_capacity: 0
  });

  // Fetch data functions
  const fetchTeams = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams({
        q: searchQuery,
        league: filters.league,
        confederation: filters.confederation,
        country: filters.country,
        founded_min: filters.founded_min,
        founded_max: filters.founded_max,
        capacity_min: filters.capacity_min,
        capacity_max: filters.capacity_max,
        division_level: filters.division_level,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: currentPage,
        per_page: itemsPerPage
      });

      const response = await fetch(`${API_BASE_URL}/api/teams?${queryParams}`);
      if (!response.ok) throw new Error('Failed to fetch teams');
      
      const data = await response.json();
      setTeams(data.teams || []);
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
        fetch(`${API_BASE_URL}/api/user/favorites/teams`)
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
        setFavorites(new Set(favoritesData.teams?.map(t => t.id) || []));
      }
    } catch (err) {
      console.error('Error fetching filter data:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchTeams();
    fetchFilterData();
  }, [searchQuery, filters, sortBy, sortOrder, currentPage]);

  // Handle favorite toggle
  const handleFavoriteToggle = async (teamId) => {
    try {
      const isFavorite = favorites.has(teamId);
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/teams/${teamId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(teamId);
        } else {
          newFavorites.add(teamId);
        }
        setFavorites(newFavorites);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Handle navigation
  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
  };

  const handleLeagueClick = (leagueId) => {
    navigate(`/leagues/${leagueId}`);
  };

  // Export functionality
  const handleExport = () => {
    const dataToExport = {
      teams: filteredTeams,
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
    link.download = `teams_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Filter and sort teams
  const filteredTeams = useMemo(() => {
    let filtered = teams.filter(team => {
      const matchesSearch = !searchQuery || 
        team.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        team.country?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        team.league?.name?.toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesSearch;
    });

    // Sort teams
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      if (sortBy === 'founded') {
        aValue = parseInt(a.founded) || 0;
        bValue = parseInt(b.founded) || 0;
      } else if (sortBy === 'capacity') {
        aValue = parseInt(a.capacity) || 0;
        bValue = parseInt(b.capacity) || 0;
      } else if (typeof aValue === 'string') {
        aValue = aValue?.toLowerCase() || '';
        bValue = bValue?.toLowerCase() || '';
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [teams, searchQuery, sortBy, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredTeams.length / itemsPerPage);
  const paginatedTeams = filteredTeams.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Statistics cards
  const statsCards = [
    {
      id: 'total_teams',
      type: 'teams',
      icon: Users,
      label: 'Total Teams',
      value: statistics.total_teams || 0,
      description: 'In database',
      change: 8.2,
      changeType: 'positive'
    },
    {
      id: 'active_teams',
      type: 'active',
      icon: Activity,
      label: 'Active Teams',
      value: statistics.active_teams || 0,
      description: 'Currently playing',
      progress: ((statistics.active_teams || 0) / (statistics.total_teams || 1)) * 100,
      progressColor: 'green'
    },
    {
      id: 'leagues_covered',
      type: 'leagues',
      icon: Trophy,
      label: 'Leagues Covered',
      value: statistics.leagues_covered || 0,
      description: 'Competitions worldwide',
      actions: [
        { 
          label: 'View All', 
          icon: Eye, 
          onClick: () => navigate('/leagues')
        }
      ]
    },
    {
      id: 'countries_covered',
      type: 'countries',
      icon: Globe,
      label: 'Countries',
      value: statistics.countries_covered || 0,
      description: 'Global coverage',
      change: 2.1,
      changeType: 'positive'
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

  // Division level options
  const divisionOptions = [
    { value: '', label: 'All Divisions' },
    { value: '1', label: '1st Division' },
    { value: '2', label: '2nd Division' },
    { value: '3', label: '3rd Division' }
  ];

  // Get team performance indicator
  const getPerformanceColor = (stats) => {
    if (!stats) return 'gray';
    const winRate = (stats.wins || 0) / (stats.matches_played || 1);
    if (winRate >= 0.7) return 'green';
    if (winRate >= 0.5) return 'blue';
    if (winRate >= 0.3) return 'yellow';
    return 'red';
  };

  // Format capacity
  const formatCapacity = (capacity) => {
    if (!capacity) return 'N/A';
    return parseInt(capacity).toLocaleString();
  };

  // Get confederation color
  const getConfederationColor = (confederation) => {
    const colorMap = {
      'UEFA': 'blue',
      'CONMEBOL': 'green',
      'CONCACAF': 'yellow',
      'AFC': 'red',
      'CAF': 'orange',
      'OFC': 'purple'
    };
    return colorMap[confederation] || 'gray';
  };

  if (loading && teams.length === 0) {
    return (
      <div className="p-6">
        <Loading context="teams" message="Loading teams database..." />
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
              Error loading teams
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchTeams}
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
              Football Teams
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Complete teams database from {leagues.length}+ leagues worldwide
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Last updated */}
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
            
            {/* Refresh button */}
            <button
              onClick={fetchTeams}
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
            if (statId === 'leagues_covered') {
              navigate('/leagues');
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
                placeholder="Search teams, countries, or leagues..."
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

                {/* Country Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Country
                  </label>
                  <select
                    value={filters.country}
                    onChange={(e) => setFilters(prev => ({ ...prev, country: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    <option value="">All Countries</option>
                    {countries.map(country => (
                      <option key={country.code} value={country.code}>
                        {country.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Division Level Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Division Level
                  </label>
                  <select
                    value={filters.division_level}
                    onChange={(e) => setFilters(prev => ({ ...prev, division_level: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {divisionOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Founded Year Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Founded Year Range
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      placeholder="Min Year"
                      value={filters.founded_min}
                      onChange={(e) => setFilters(prev => ({ ...prev, founded_min: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                    <input
                      type="number"
                      placeholder="Max Year"
                      value={filters.founded_max}
                      onChange={(e) => setFilters(prev => ({ ...prev, founded_max: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                  </div>
                </div>

                {/* Capacity Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Stadium Capacity Range
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      placeholder="Min"
                      value={filters.capacity_min}
                      onChange={(e) => setFilters(prev => ({ ...prev, capacity_min: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                    <input
                      type="number"
                      placeholder="Max"
                      value={filters.capacity_max}
                      onChange={(e) => setFilters(prev => ({ ...prev, capacity_max: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                  </div>
                </div>
              </div>
              
              {/* Clear Filters */}
              <div className="flex justify-end">
                <button
                  onClick={() => setFilters({
                    league: '',
                    confederation: '',
                    country: '',
                    founded_min: '',
                    founded_max: '',
                    capacity_min: '',
                    capacity_max: '',
                    division_level: ''
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
            Showing {paginatedTeams.length} of {filteredTeams.length} teams
          </div>
          
          {/* Sort Controls */}
          <div className="flex items-center space-x-4">
            <label className="font-medium">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="name">Name</option>
              <option value="country">Country</option>
              <option value="founded">Founded</option>
              <option value="capacity">Capacity</option>
              <option value="league">League</option>
            </select>
            
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Teams Display */}
        {viewMode === 'cards' ? (
          /* Cards View */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {paginatedTeams.map(team => (
              <div
                key={team.id}
                className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow cursor-pointer group"
                onClick={() => handleTeamClick(team.id)}
              >
                {/* Team Header */}
                <div className="p-4 pb-2">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-3 flex-1">
                      <img
                        src={team.logo || '/api/placeholder/48/48'}
                        alt={team.name}
                        className="w-12 h-12 rounded-lg"
                      />
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors truncate">
                          {team.name}
                        </h3>
                        <div className="flex items-center space-x-2 mt-1">
                          <Flag className="w-3 h-3 text-gray-400" />
                          <span className="text-sm text-gray-500 dark:text-gray-400 truncate">
                            {team.country}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleFavoriteToggle(team.id);
                      }}
                      className="text-gray-400 hover:text-yellow-500 transition-colors flex-shrink-0"
                    >
                      {favorites.has(team.id) ? (
                        <BookmarkCheck className="w-5 h-5 text-yellow-500" />
                      ) : (
                        <Bookmark className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* League & Division */}
                <div className="px-4 pb-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLeagueClick(team.league?.id);
                    }}
                    className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors w-full"
                  >
                    <img
                      src={team.league?.logo || '/api/placeholder/16/16'}
                      alt={team.league?.name}
                      className="w-4 h-4 rounded"
                    />
                    <span className="text-sm font-medium truncate">{team.league?.name}</span>
                  </button>
                  
                  {team.league?.level && (
                    <div className="flex items-center space-x-2 mt-1">
                      <div className={`w-2 h-2 rounded-full bg-${getConfederationColor(team.league?.confederation)}-500`}></div>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {team.league.level === 1 ? '1st Division' : 
                         team.league.level === 2 ? '2nd Division' : 
                         team.league.level === 3 ? '3rd Division' : 'Division ' + team.league.level}
                      </span>
                    </div>
                  )}
                </div>

                {/* Stadium Info */}
                {team.venue && (
                  <div className="px-4 pb-2">
                    <div className="flex items-center space-x-2">
                      <Building className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400 truncate">
                        {team.venue}
                      </span>
                    </div>
                    {team.capacity && (
                      <div className="flex items-center space-x-2 mt-1">
                        <Users className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {formatCapacity(team.capacity)} capacity
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Team Stats */}
                {team.current_season_stats && (
                  <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700/50">
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div>
                        <div className="text-lg font-bold text-gray-900 dark:text-white">
                          {team.current_season_stats.wins || 0}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Wins</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold text-gray-900 dark:text-white">
                          {team.current_season_stats.draws || 0}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Draws</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold text-gray-900 dark:text-white">
                          {team.current_season_stats.losses || 0}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">Losses</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Team Info Footer */}
                <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Founded: {team.founded || 'Unknown'}
                      </span>
                    </div>
                    
                    {team.current_season_stats?.league_position && (
                      <div className="flex items-center space-x-1">
                        <Award className="w-4 h-4 text-yellow-500" />
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          #{team.current_season_stats.league_position}
                        </span>
                      </div>
                    )}
                  </div>
                  
                  {/* Current Form */}
                  {team.current_season_stats?.form && (
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-xs text-gray-500 dark:text-gray-400">Recent Form:</span>
                      <div className="flex space-x-1">
                        {team.current_season_stats.form.split('').slice(-5).map((result, index) => (
                          <div
                            key={index}
                            className={`w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold text-white ${
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
                      Team
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      League
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Country
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Founded
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Capacity
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Position
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Form
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {paginatedTeams.map(team => (
                    <tr
                      key={team.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                      onClick={() => handleTeamClick(team.id)}
                    >
                      {/* Team */}
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-3">
                          <img
                            src={team.logo || '/api/placeholder/32/32'}
                            alt={team.name}
                            className="w-8 h-8 rounded"
                          />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {team.name}
                            </div>
                            {team.venue && (
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                {team.venue}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>

                      {/* League */}
                      <td className="px-6 py-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleLeagueClick(team.league?.id);
                          }}
                          className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                        >
                          <img
                            src={team.league?.logo || '/api/placeholder/20/20'}
                            alt={team.league?.name}
                            className="w-5 h-5 rounded"
                          />
                          <span className="font-medium">{team.league?.name}</span>
                        </button>
                      </td>

                      {/* Country */}
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-2">
                          <Flag className="w-4 h-4 text-gray-400" />
                          <span className="font-medium text-gray-900 dark:text-white">
                            {team.country}
                          </span>
                        </div>
                      </td>

                      {/* Founded */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {team.founded || 'Unknown'}
                        </span>
                      </td>

                      {/* Capacity */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {formatCapacity(team.capacity)}
                        </span>
                      </td>

                      {/* League Position */}
                      <td className="px-6 py-4 text-center">
                        {team.current_season_stats?.league_position ? (
                          <div className="flex items-center justify-center space-x-1">
                            <Award className="w-4 h-4 text-yellow-500" />
                            <span className="font-medium text-gray-900 dark:text-white">
                              #{team.current_season_stats.league_position}
                            </span>
                          </div>
                        ) : (
                          <span className="text-gray-400">N/A</span>
                        )}
                      </td>

                      {/* Form */}
                      <td className="px-6 py-4 text-center">
                        {team.current_season_stats?.form ? (
                          <div className="flex justify-center space-x-1">
                            {team.current_season_stats.form.split('').slice(-5).map((result, index) => (
                              <div
                                key={index}
                                className={`w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                                  result === 'W' ? 'bg-green-500' :
                                  result === 'D' ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                              >
                                {result}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span className="text-gray-400">N/A</span>
                        )}
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleFavoriteToggle(team.id);
                            }}
                            className="text-gray-400 hover:text-yellow-500 transition-colors"
                          >
                            {favorites.has(team.id) ? (
                              <BookmarkCheck className="w-4 h-4 text-yellow-500" />
                            ) : (
                              <Bookmark className="w-4 h-4" />
                            )}
                          </button>
                          
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleTeamClick(team.id);
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
        {filteredTeams.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <Users className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No teams found
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              Try adjusting your search criteria or filters
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters({
                  league: '',
                  confederation: '',
                  country: '',
                  founded_min: '',
                  founded_max: '',
                  capacity_min: '',
                  capacity_max: '',
                  division_level: ''
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

export default TeamsList;