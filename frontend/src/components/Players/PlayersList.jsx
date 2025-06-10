import {
    Activity,
    AlertTriangle,
    Bookmark,
    BookmarkCheck,
    Calendar,
    ChevronDown,
    ChevronUp,
    Download,
    Eye,
    Filter,
    Flag,
    Grid,
    Heart,
    List,
    RefreshCw,
    Search,
    SortAsc,
    SortDesc,
    Star,
    Users
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const PlayersList = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State Management
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Filter States
  const [filters, setFilters] = useState({
    position: '', // GK, DEF, MID, FWD
    team: '',
    league: '',
    confederation: '',
    country: '',
    age_min: '',
    age_max: '',
    market_value_min: '',
    market_value_max: '',
    availability: 'all' // all, available, injured
  });
  
  // UI States
  const [viewMode, setViewMode] = useState('cards'); // cards, table, grid
  const [sortBy, setSortBy] = useState('market_value');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(24);
  const [showFilters, setShowFilters] = useState(false);
  
  // Data for filters
  const [teams, setTeams] = useState([]);
  const [leagues, setLeagues] = useState([]);
  const [countries, setCountries] = useState([]);
  const [favorites, setFavorites] = useState(new Set());
  
  // Statistics
  const [statistics, setStatistics] = useState({
    total_players: 0,
    active_players: 0,
    injured_players: 0,
    avg_age: 0,
    avg_market_value: 0
  });

  // Fetch data functions
  const fetchPlayers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams({
        q: searchQuery,
        position: filters.position,
        team: filters.team,
        league: filters.league,
        confederation: filters.confederation,
        country: filters.country,
        age_min: filters.age_min,
        age_max: filters.age_max,
        market_value_min: filters.market_value_min,
        market_value_max: filters.market_value_max,
        availability: filters.availability,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: currentPage,
        per_page: itemsPerPage
      });

      const response = await fetch(`${API_BASE_URL}/api/players?${queryParams}`);
      if (!response.ok) throw new Error('Failed to fetch players');
      
      const data = await response.json();
      setPlayers(data.players || []);
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
      const [teamsRes, leaguesRes, countriesRes, favoritesRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/teams`),
        fetch(`${API_BASE_URL}/api/leagues`),
        fetch(`${API_BASE_URL}/api/countries`),
        fetch(`${API_BASE_URL}/api/user/favorites/players`)
      ]);

      if (teamsRes.ok) {
        const teamsData = await teamsRes.json();
        setTeams(teamsData.teams || []);
      }

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
        setFavorites(new Set(favoritesData.players?.map(p => p.id) || []));
      }
    } catch (err) {
      console.error('Error fetching filter data:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchPlayers();
    fetchFilterData();
  }, [searchQuery, filters, sortBy, sortOrder, currentPage]);

  // Handle favorite toggle
  const handleFavoriteToggle = async (playerId) => {
    try {
      const isFavorite = favorites.has(playerId);
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/api/user/favorites/players/${playerId}`, {
        method
      });

      if (response.ok) {
        const newFavorites = new Set(favorites);
        if (isFavorite) {
          newFavorites.delete(playerId);
        } else {
          newFavorites.add(playerId);
        }
        setFavorites(newFavorites);
      }
    } catch (err) {
      console.error('Error toggling favorite:', err);
    }
  };

  // Handle navigation
  const handlePlayerClick = (playerId) => {
    navigate(`/players/${playerId}`);
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
      players: filteredPlayers,
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
    link.download = `players_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Filter and sort players
  const filteredPlayers = useMemo(() => {
    let filtered = players.filter(player => {
      const matchesSearch = !searchQuery || 
        player.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        player.team?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        player.nationality?.toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesSearch;
    });

    // Sort players
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      if (sortBy === 'age') {
        aValue = calculateAge(a.date_of_birth);
        bValue = calculateAge(b.date_of_birth);
      } else if (sortBy === 'market_value') {
        aValue = parseFloat(a.market_value?.replace(/[^0-9.]/g, '') || 0);
        bValue = parseFloat(b.market_value?.replace(/[^0-9.]/g, '') || 0);
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [players, searchQuery, sortBy, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredPlayers.length / itemsPerPage);
  const paginatedPlayers = filteredPlayers.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

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

  // Statistics cards
  const statsCards = [
    {
      id: 'total_players',
      type: 'players',
      icon: Users,
      label: 'Total Players',
      value: statistics.total_players || 0,
      description: 'In database',
      change: 15.2,
      changeType: 'positive'
    },
    {
      id: 'active_players',
      type: 'active',
      icon: Activity,
      label: 'Active Players',
      value: statistics.active_players || 0,
      description: 'Currently playing',
      progress: ((statistics.active_players || 0) / (statistics.total_players || 1)) * 100,
      progressColor: 'green'
    },
    {
      id: 'injured_players',
      type: 'injured',
      icon: Heart,
      label: 'Injured Players',
      value: statistics.injured_players || 0,
      description: 'Currently injured',
      progress: ((statistics.injured_players || 0) / (statistics.total_players || 1)) * 100,
      progressColor: 'red',
      actions: [
        { 
          label: 'View All', 
          icon: Eye, 
          onClick: () => setFilters(prev => ({ ...prev, availability: 'injured' }))
        }
      ]
    },
    {
      id: 'avg_age',
      type: 'demographics',
      icon: Calendar,
      label: 'Average Age',
      value: `${(statistics.avg_age || 0).toFixed(1)} years`,
      description: 'Player demographics',
      change: -0.3,
      changeType: 'negative'
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

  // Position options
  const positionOptions = [
    { value: '', label: 'All Positions', color: 'gray' },
    { value: 'GK', label: 'Goalkeeper', color: 'yellow' },
    { value: 'DEF', label: 'Defender', color: 'blue' },
    { value: 'MID', label: 'Midfielder', color: 'green' },
    { value: 'FWD', label: 'Forward', color: 'red' }
  ];

  // Availability options
  const availabilityOptions = [
    { value: 'all', label: 'All Players' },
    { value: 'available', label: 'Available' },
    { value: 'injured', label: 'Injured' }
  ];

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

  // Format market value
  const formatMarketValue = (value) => {
    if (!value) return 'N/A';
    const numValue = parseFloat(value.replace(/[^0-9.]/g, ''));
    if (numValue >= 1000000) {
      return `€${(numValue / 1000000).toFixed(1)}M`;
    } else if (numValue >= 1000) {
      return `€${(numValue / 1000).toFixed(0)}K`;
    }
    return `€${numValue}`;
  };

  if (loading && players.length === 0) {
    return (
      <div className="p-6">
        <Loading context="players" message="Loading players database..." />
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
              Error loading players
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchPlayers}
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
              Football Players
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Comprehensive player database from {leagues.length}+ leagues worldwide
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Last updated */}
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
            
            {/* Refresh button */}
            <button
              onClick={fetchPlayers}
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
            if (statId === 'injured_players') {
              setFilters(prev => ({ ...prev, availability: 'injured' }));
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
                placeholder="Search players, teams, or nationalities..."
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
                {/* Position Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Position
                  </label>
                  <select
                    value={filters.position}
                    onChange={(e) => setFilters(prev => ({ ...prev, position: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {positionOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Availability Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Availability
                  </label>
                  <select
                    value={filters.availability}
                    onChange={(e) => setFilters(prev => ({ ...prev, availability: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {availabilityOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Age Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Age Range
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      placeholder="Min"
                      value={filters.age_min}
                      onChange={(e) => setFilters(prev => ({ ...prev, age_min: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                    <input
                      type="number"
                      placeholder="Max"
                      value={filters.age_max}
                      onChange={(e) => setFilters(prev => ({ ...prev, age_max: e.target.value }))}
                      className="w-1/2 border border-gray-300 rounded-lg px-2 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                  </div>
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
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Team Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Team
                  </label>
                  <select
                    value={filters.team}
                    onChange={(e) => setFilters(prev => ({ ...prev, team: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    <option value="">All Teams</option>
                    {teams.map(team => (
                      <option key={team.id} value={team.id}>
                        {team.name}
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

                {/* Country Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Nationality
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
              </div>
              
              {/* Clear Filters */}
              <div className="flex justify-end">
                <button
                  onClick={() => setFilters({
                    position: '',
                    team: '',
                    league: '',
                    confederation: '',
                    country: '',
                    age_min: '',
                    age_max: '',
                    market_value_min: '',
                    market_value_max: '',
                    availability: 'all'
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
            Showing {paginatedPlayers.length} of {filteredPlayers.length} players
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
              <option value="age">Age</option>
              <option value="position">Position</option>
              <option value="market_value">Market Value</option>
              <option value="rating">Rating</option>
            </select>
            
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Players Display */}
        {viewMode === 'cards' ? (
          /* Cards View */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {paginatedPlayers.map(player => (
              <div
                key={player.id}
                className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow cursor-pointer group"
                onClick={() => handlePlayerClick(player.id)}
              >
                {/* Player Header */}
                <div className="relative p-4 pb-2">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-3">
                      <img
                        src={player.photo || '/api/placeholder/48/48'}
                        alt={player.name}
                        className="w-12 h-12 rounded-full"
                      />
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                          {player.name}
                        </h3>
                        <div className="flex items-center space-x-2 mt-1">
                          <div className={`w-2 h-2 rounded-full bg-${getPositionColor(player.position)}-500`}></div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {player.position}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">•</span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {calculateAge(player.date_of_birth)}y
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleFavoriteToggle(player.id);
                      }}
                      className="text-gray-400 hover:text-yellow-500 transition-colors"
                    >
                      {favorites.has(player.id) ? (
                        <BookmarkCheck className="w-5 h-5 text-yellow-500" />
                      ) : (
                        <Bookmark className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                  
                  {/* Injury Alert */}
                  {player.is_injured && (
                    <div className="absolute top-2 right-2">
                      <div className="bg-red-100 dark:bg-red-900/20 p-1 rounded-full">
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Team & League */}
                <div className="px-4 pb-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTeamClick(player.team?.id);
                    }}
                    className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                  >
                    <img
                      src={player.team?.logo || '/api/placeholder/16/16'}
                      alt={player.team?.name}
                      className="w-4 h-4 rounded"
                    />
                    <span className="text-sm font-medium">{player.team?.name}</span>
                  </button>
                  
                  {player.league && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleLeagueClick(player.league?.id);
                      }}
                      className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 mt-1 block"
                    >
                      {player.league.name}
                    </button>
                  )}
                </div>

                {/* Player Stats */}
                <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700/50">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div>
                      <div className="text-lg font-bold text-gray-900 dark:text-white">
                        {player.season_goals || 0}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Goals</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold text-gray-900 dark:text-white">
                        {player.season_assists || 0}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Assists</div>
                    </div>
                    <div>
                      <div className="text-lg font-bold text-gray-900 dark:text-white">
                        {(player.average_rating || 0).toFixed(1)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Rating</div>
                    </div>
                  </div>
                </div>

                {/* Player Value & Nationality */}
                <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Flag className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {player.nationality}
                      </span>
                    </div>
                    
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {formatMarketValue(player.market_value)}
                    </div>
                  </div>
                  
                  {player.jersey_number && (
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        Jersey #{player.jersey_number}
                      </span>
                      
                      {player.international_caps && (
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {player.international_caps} caps
                        </span>
                      )}
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
                      Player
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Position
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Team
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Age
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Goals
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Assists
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Rating
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Value
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {paginatedPlayers.map(player => (
                    <tr
                      key={player.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                      onClick={() => handlePlayerClick(player.id)}
                    >
                      {/* Player */}
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-3">
                          <img
                            src={player.photo || '/api/placeholder/32/32'}
                            alt={player.name}
                            className="w-8 h-8 rounded-full"
                          />
                          <div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {player.name}
                            </div>
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {player.nationality}
                            </div>
                          </div>
                          {player.is_injured && (
                            <AlertTriangle className="w-4 h-4 text-red-500" />
                          )}
                        </div>
                      </td>

                      {/* Position */}
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-2">
                          <div className={`w-3 h-3 rounded-full bg-${getPositionColor(player.position)}-500`}></div>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {player.position}
                          </span>
                        </div>
                      </td>

                      {/* Team */}
                      <td className="px-6 py-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleTeamClick(player.team?.id);
                          }}
                          className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                        >
                          <img
                            src={player.team?.logo || '/api/placeholder/20/20'}
                            alt={player.team?.name}
                            className="w-5 h-5 rounded"
                          />
                          <span className="font-medium">{player.team?.name}</span>
                        </button>
                      </td>

                      {/* Age */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {calculateAge(player.date_of_birth)}
                        </span>
                      </td>

                      {/* Goals */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {player.season_goals || 0}
                        </span>
                      </td>

                      {/* Assists */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {player.season_assists || 0}
                        </span>
                      </td>

                      {/* Rating */}
                      <td className="px-6 py-4 text-center">
                        <div className="flex items-center justify-center space-x-1">
                          <Star className="w-4 h-4 text-yellow-500" />
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(player.average_rating || 0).toFixed(1)}
                          </span>
                        </div>
                      </td>

                      {/* Market Value */}
                      <td className="px-6 py-4 text-center">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {formatMarketValue(player.market_value)}
                        </span>
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleFavoriteToggle(player.id);
                            }}
                            className="text-gray-400 hover:text-yellow-500 transition-colors"
                          >
                            {favorites.has(player.id) ? (
                              <BookmarkCheck className="w-4 h-4 text-yellow-500" />
                            ) : (
                              <Bookmark className="w-4 h-4" />
                            )}
                          </button>
                          
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handlePlayerClick(player.id);
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
        {filteredPlayers.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <Users className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No players found
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              Try adjusting your search criteria or filters
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters({
                  position: '',
                  team: '',
                  league: '',
                  confederation: '',
                  country: '',
                  age_min: '',
                  age_max: '',
                  market_value_min: '',
                  market_value_max: '',
                  availability: 'all'
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

export default PlayersList;