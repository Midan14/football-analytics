import {
    Activity,
    ArrowUpDown,
    Award,
    Bookmark,
    BookmarkCheck,
    Download,
    ExternalLink,
    Eye,
    Filter,
    Flag,
    Globe,
    Grid,
    List,
    Search,
    Star,
    Trophy,
    Users
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

// Import components
import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const LeaguesList = () => {
  // Hooks
  const navigate = useNavigate();
  const location = useLocation();

  // Estados principales
  const [leagues, setLeagues] = useState([]);
  const [filteredLeagues, setFilteredLeagues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [favorites, setFavorites] = useState([]);

  // Estados de filtros
  const [filters, setFilters] = useState({
    search: '',
    confederation: '',
    country: '',
    level: '',
    gender: '',
    is_active: true
  });

  // Estados de configuración
  const [viewMode, setViewMode] = useState('cards'); // cards, table
  const [sortBy, setSortBy] = useState('name'); // name, country, level, teams_count
  const [sortOrder, setSortOrder] = useState('asc'); // asc, desc
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 24,
    total: 0
  });

  // Estados de UI
  const [showFilters, setShowFilters] = useState(false);
  const [selectedLeagues, setSelectedLeagues] = useState([]);

  // Datos para filtros
  const [countries, setCountries] = useState([]);
  const [leagueStats, setLeagueStats] = useState(null);

  const confederations = [
    { code: 'UEFA', name: 'UEFA (Europe)', color: 'blue' },
    { code: 'CONMEBOL', name: 'CONMEBOL (South America)', color: 'green' },
    { code: 'CONCACAF', name: 'CONCACAF (North/Central America)', color: 'yellow' },
    { code: 'AFC', name: 'AFC (Asia)', color: 'red' },
    { code: 'CAF', name: 'CAF (Africa)', color: 'orange' },
    { code: 'OFC', name: 'OFC (Oceania)', color: 'purple' }
  ];

  const levels = [
    { value: '1', label: '1st Division' },
    { value: '2', label: '2nd Division' },
    { value: '3', label: '3rd Division' }
  ];

  // Efectos
  useEffect(() => {
    fetchLeagues();
    fetchCountries();
    fetchLeagueStats();
    fetchFavorites();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [leagues, filters, sortBy, sortOrder]);

  useEffect(() => {
    // Handle URL query parameters
    const params = new URLSearchParams(location.search);
    const urlFilters = {};
    
    ['confederation', 'country', 'level', 'gender'].forEach(key => {
      const value = params.get(key);
      if (value) urlFilters[key] = value;
    });
    
    if (Object.keys(urlFilters).length > 0) {
      setFilters(prev => ({ ...prev, ...urlFilters }));
    }
  }, [location.search]);

  // API Functions
  const fetchLeagues = async () => {
    setLoading(true);
    setError(null);

    try {
      const queryParams = new URLSearchParams({
        ...filters,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: pagination.page,
        limit: pagination.limit,
        include_stats: true
      }).toString();

      const response = await fetch(`${API_BASE_URL}/leagues?${queryParams}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch leagues');
      }

      const data = await response.json();
      setLeagues(data.leagues || []);
      setPagination(prev => ({
        ...prev,
        total: data.pagination?.total || 0
      }));

    } catch (error) {
      console.error('Error fetching leagues:', error);
      setError('Failed to load leagues. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchCountries = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/leagues/countries`);
      if (response.ok) {
        const data = await response.json();
        setCountries(data.countries || []);
      }
    } catch (error) {
      console.error('Error fetching countries:', error);
    }
  };

  const fetchLeagueStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/leagues/statistics`);
      if (response.ok) {
        const data = await response.json();
        setLeagueStats(data.stats);
      }
    } catch (error) {
      console.error('Error fetching league stats:', error);
    }
  };

  const fetchFavorites = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/favorites/leagues`);
      if (response.ok) {
        const data = await response.json();
        setFavorites(data.leagues?.map(l => l.id) || []);
      }
    } catch (error) {
      console.error('Error fetching favorites:', error);
    }
  };

  const toggleFavorite = async (leagueId) => {
    try {
      const isFavorite = favorites.includes(leagueId);
      const method = isFavorite ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE_URL}/user/favorites/leagues/${leagueId}`, {
        method
      });
      
      if (response.ok) {
        setFavorites(prev => 
          isFavorite 
            ? prev.filter(id => id !== leagueId)
            : [...prev, leagueId]
        );
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  // Utility functions
  const applyFilters = () => {
    let filtered = [...leagues];

    // Apply search filter
    if (filters.search) {
      const searchTerm = filters.search.toLowerCase();
      filtered = filtered.filter(league =>
        league.name.toLowerCase().includes(searchTerm) ||
        league.country.toLowerCase().includes(searchTerm) ||
        league.code.toLowerCase().includes(searchTerm)
      );
    }

    // Apply other filters
    Object.keys(filters).forEach(key => {
      if (filters[key] && key !== 'search') {
        filtered = filtered.filter(league => {
          if (key === 'is_active') return league[key] === filters[key];
          return league[key] === filters[key];
        });
      }
    });

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];

      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });

    setFilteredLeagues(filtered);
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPagination(prev => ({ ...prev, page: 1 }));
  };

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  const handleLeagueClick = (leagueId) => {
    navigate(`/leagues/${leagueId}`);
  };

  const clearFilters = () => {
    setFilters({
      search: '',
      confederation: '',
      country: '',
      level: '',
      gender: '',
      is_active: true
    });
  };

  const exportLeagues = () => {
    const data = {
      leagues: filteredLeagues,
      filters,
      total: filteredLeagues.length,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `football-leagues-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Stats cards data
  const statsCardsData = useMemo(() => {
    if (!leagueStats) return [];

    return [
      {
        id: 'total_leagues',
        icon: Trophy,
        label: 'Total Leagues',
        value: leagueStats.total_leagues || 265,
        description: 'Worldwide coverage',
        type: 'leagues'
      },
      {
        id: 'active_leagues',
        icon: Activity,
        label: 'Active Leagues',
        value: leagueStats.active_leagues || 0,
        description: 'Currently playing',
        type: 'live'
      },
      {
        id: 'confederations',
        icon: Globe,
        label: 'Confederations',
        value: 6,
        description: 'Global coverage',
        type: 'analytics'
      },
      {
        id: 'countries',
        icon: Flag,
        label: 'Countries',
        value: countries.length,
        description: 'Nations covered',
        type: 'teams'
      }
    ];
  }, [leagueStats, countries]);

  if (loading && leagues.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <Loading type="card" context="leagues" size="large" message="Loading leagues from around the world..." />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-4xl font-bold text-gray-900 mb-2 flex items-center gap-3">
                  <Trophy className="text-blue-600" />
                  Football Leagues
                </h1>
                <p className="text-gray-600 text-lg">
                  Explore {filteredLeagues.length} leagues from {confederations.length} confederations worldwide
                </p>
              </div>
              
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                    showFilters 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Filter size={16} />
                  Filters
                </button>
                
                <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                  <button
                    onClick={() => setViewMode('cards')}
                    className={`px-3 py-2 transition-colors ${
                      viewMode === 'cards'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <Grid size={16} />
                  </button>
                  <button
                    onClick={() => setViewMode('table')}
                    className={`px-3 py-2 transition-colors ${
                      viewMode === 'table'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <List size={16} />
                  </button>
                </div>
                
                <button
                  onClick={exportLeagues}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center gap-2 transition-colors"
                >
                  <Download size={16} />
                  Export
                </button>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          {leagueStats && (
            <div className="mb-8">
              <StatsCards 
                stats={statsCardsData}
                loading={false}
                cardSize="medium"
                columns={4}
              />
            </div>
          )}

          {/* Filters Panel */}
          {showFilters && (
            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
                {/* Search */}
                <div className="lg:col-span-2">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Search Leagues
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                    <input
                      type="text"
                      placeholder="Search by name, country, or code..."
                      value={filters.search}
                      onChange={(e) => handleFilterChange('search', e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>

                {/* Confederation */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Confederation
                  </label>
                  <select
                    value={filters.confederation}
                    onChange={(e) => handleFilterChange('confederation', e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All Confederations</option>
                    {confederations.map(conf => (
                      <option key={conf.code} value={conf.code}>{conf.name}</option>
                    ))}
                  </select>
                </div>

                {/* Country */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Country
                  </label>
                  <select
                    value={filters.country}
                    onChange={(e) => handleFilterChange('country', e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All Countries</option>
                    {countries.map(country => (
                      <option key={country} value={country}>{country}</option>
                    ))}
                  </select>
                </div>

                {/* Level */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Division Level
                  </label>
                  <select
                    value={filters.level}
                    onChange={(e) => handleFilterChange('level', e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All Levels</option>
                    {levels.map(level => (
                      <option key={level.value} value={level.value}>{level.label}</option>
                    ))}
                  </select>
                </div>

                {/* Gender */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Gender
                  </label>
                  <select
                    value={filters.gender}
                    onChange={(e) => handleFilterChange('gender', e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All Genders</option>
                    <option value="M">Men's</option>
                    <option value="F">Women's</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
                <div className="text-sm text-gray-600">
                  Showing {filteredLeagues.length} of {leagues.length} leagues
                </div>
                <button
                  onClick={clearFilters}
                  className="px-4 py-2 text-blue-600 hover:text-blue-700 transition-colors"
                >
                  Clear All Filters
                </button>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg mb-6">
              <div className="flex items-center gap-2">
                <Trophy className="text-red-600" size={20} />
                <span className="font-semibold">Error Loading Leagues</span>
              </div>
              <p className="mt-1">{error}</p>
              <button
                onClick={fetchLeagues}
                className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {/* Content */}
          {viewMode === 'cards' ? (
            // Cards View
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredLeagues.map((league) => (
                <div
                  key={league.id}
                  onClick={() => handleLeagueClick(league.id)}
                  className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer group overflow-hidden"
                >
                  {/* Card Header */}
                  <div className="p-6 pb-4">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-${
                          confederations.find(c => c.code === league.confederation)?.color || 'blue'
                        }-100`}>
                          <Trophy size={24} className={`text-${
                            confederations.find(c => c.code === league.confederation)?.color || 'blue'
                          }-600`} />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-bold text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
                            {league.name}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            {league.level === 1 && <Star size={14} className="text-yellow-500" />}
                            <span className="text-sm text-gray-500">{league.code}</span>
                          </div>
                        </div>
                      </div>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(league.id);
                        }}
                        className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                      >
                        {favorites.includes(league.id) ? (
                          <BookmarkCheck size={16} className="text-yellow-500" />
                        ) : (
                          <Bookmark size={16} className="text-gray-400" />
                        )}
                      </button>
                    </div>

                    {/* League Info */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Flag size={14} />
                        <span>{league.country}</span>
                        <span className="text-gray-400">•</span>
                        <span>{league.confederation}</span>
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm">
                        <div className="flex items-center gap-1 text-gray-600">
                          <Award size={14} />
                          <span>Level {league.level}</span>
                        </div>
                        
                        {league.teams_count && (
                          <div className="flex items-center gap-1 text-gray-600">
                            <Users size={14} />
                            <span>{league.teams_count} teams</span>
                          </div>
                        )}
                      </div>

                      {league.gender === 'F' && (
                        <div className="inline-block px-2 py-1 bg-pink-100 text-pink-700 rounded-full text-xs font-medium">
                          Women's League
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Card Stats */}
                  {league.statistics && (
                    <div className="px-6 py-4 bg-gray-50 border-t border-gray-100">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="text-center">
                          <div className="font-semibold text-blue-600">
                            {league.statistics.total_matches || 0}
                          </div>
                          <div className="text-gray-500">Matches</div>
                        </div>
                        <div className="text-center">
                          <div className="font-semibold text-green-600">
                            {league.statistics.avg_goals_per_match?.toFixed(1) || 'N/A'}
                          </div>
                          <div className="text-gray-500">Avg Goals</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Card Footer */}
                  <div className="px-6 py-3 bg-gray-50 border-t border-gray-100">
                    <div className="flex items-center justify-between">
                      <div className={`w-2 h-2 rounded-full ${
                        league.is_active ? 'bg-green-500' : 'bg-gray-400'
                      }`}></div>
                      <span className="text-xs text-gray-500">
                        {league.is_active ? 'Active' : 'Inactive'}
                      </span>
                      <ExternalLink size={14} className="text-gray-400 group-hover:text-blue-600 transition-colors" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            // Table View
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th className="text-left py-4 px-6">
                        <button
                          onClick={() => handleSort('name')}
                          className="flex items-center gap-2 font-semibold text-gray-700 hover:text-gray-900"
                        >
                          League
                          <ArrowUpDown size={14} />
                        </button>
                      </th>
                      <th className="text-left py-4 px-6">
                        <button
                          onClick={() => handleSort('country')}
                          className="flex items-center gap-2 font-semibold text-gray-700 hover:text-gray-900"
                        >
                          Country
                          <ArrowUpDown size={14} />
                        </button>
                      </th>
                      <th className="text-left py-4 px-6">Confederation</th>
                      <th className="text-center py-4 px-6">
                        <button
                          onClick={() => handleSort('level')}
                          className="flex items-center gap-2 font-semibold text-gray-700 hover:text-gray-900"
                        >
                          Level
                          <ArrowUpDown size={14} />
                        </button>
                      </th>
                      <th className="text-center py-4 px-6">Teams</th>
                      <th className="text-center py-4 px-6">Status</th>
                      <th className="text-center py-4 px-6">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLeagues.map((league) => (
                      <tr 
                        key={league.id} 
                        className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                        onClick={() => handleLeagueClick(league.id)}
                      >
                        <td className="py-4 px-6">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center bg-${
                              confederations.find(c => c.code === league.confederation)?.color || 'blue'
                            }-100`}>
                              <Trophy size={16} className={`text-${
                                confederations.find(c => c.code === league.confederation)?.color || 'blue'
                              }-600`} />
                            </div>
                            <div>
                              <div className="font-semibold text-gray-900 flex items-center gap-2">
                                {league.name}
                                {league.level === 1 && <Star size={14} className="text-yellow-500" />}
                                {league.gender === 'F' && (
                                  <span className="px-1.5 py-0.5 bg-pink-100 text-pink-700 rounded text-xs">F</span>
                                )}
                              </div>
                              <div className="text-sm text-gray-500">{league.code}</div>
                            </div>
                          </div>
                        </td>
                        <td className="py-4 px-6">
                          <div className="flex items-center gap-2">
                            <Flag size={14} className="text-gray-400" />
                            <span className="text-gray-900">{league.country}</span>
                          </div>
                        </td>
                        <td className="py-4 px-6">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium bg-${
                            confederations.find(c => c.code === league.confederation)?.color || 'blue'
                          }-100 text-${
                            confederations.find(c => c.code === league.confederation)?.color || 'blue'
                          }-700`}>
                            {league.confederation}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-center">
                          <span className="font-medium">{league.level}</span>
                        </td>
                        <td className="py-4 px-6 text-center">
                          <span className="font-medium">{league.teams_count || 'N/A'}</span>
                        </td>
                        <td className="py-4 px-6 text-center">
                          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                            league.is_active 
                              ? 'bg-green-100 text-green-700' 
                              : 'bg-gray-100 text-gray-700'
                          }`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${
                              league.is_active ? 'bg-green-500' : 'bg-gray-400'
                            }`}></div>
                            {league.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-center">
                          <div className="flex items-center justify-center gap-2">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleFavorite(league.id);
                              }}
                              className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                            >
                              {favorites.includes(league.id) ? (
                                <BookmarkCheck size={14} className="text-yellow-500" />
                              ) : (
                                <Bookmark size={14} className="text-gray-400" />
                              )}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleLeagueClick(league.id);
                              }}
                              className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                            >
                              <Eye size={14} className="text-gray-400" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Empty State */}
              {filteredLeagues.length === 0 && !loading && (
                <div className="text-center py-12">
                  <Trophy size={48} className="mx-auto text-gray-300 mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No leagues found</h3>
                  <p className="text-gray-600 mb-4">
                    Try adjusting your filters or search terms
                  </p>
                  <button
                    onClick={clearFilters}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Clear Filters
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Pagination */}
          {filteredLeagues.length > pagination.limit && (
            <div className="mt-8 flex items-center justify-center">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPagination(prev => ({ ...prev, page: Math.max(1, prev.page - 1) }))}
                  disabled={pagination.page === 1}
                  className="px-3 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                >
                  Previous
                </button>
                
                <span className="px-4 py-2 text-sm text-gray-600">
                  Page {pagination.page} of {Math.ceil(filteredLeagues.length / pagination.limit)}
                </span>
                
                <button
                  onClick={() => setPagination(prev => ({ ...prev, page: prev.page + 1 }))}
                  disabled={pagination.page >= Math.ceil(filteredLeagues.length / pagination.limit)}
                  className="px-3 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default LeaguesList;