import {
    Activity,
    AlertTriangle,
    BarChart3,
    ChevronDown,
    ChevronUp,
    Clock,
    Download,
    Eye,
    Filter,
    Grid,
    Heart,
    List,
    RefreshCw,
    Search,
    Shield,
    SortAsc,
    SortDesc,
    Timer,
    TrendingUp,
    Users,
    Zap
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const InjuriesList = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State Management
  const [injuries, setInjuries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Filter States
  const [filters, setFilters] = useState({
    status: 'all', // all, active, recovered, pending
    severity: '', // minor, moderate, major, career_ending
    injury_type: '', // muscle, ligament, bone, head, etc.
    team: '',
    league: '',
    confederation: '',
    position: '', // GK, DEF, MID, FWD
    timeframe: 'all' // all, week, month, season
  });
  
  // UI States
  const [viewMode, setViewMode] = useState('cards'); // cards, table, timeline
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);
  const [showFilters, setShowFilters] = useState(false);
  
  // Data for filters
  const [teams, setTeams] = useState([]);
  const [leagues, setLeagues] = useState([]);
  const [injuryTypes, setInjuryTypes] = useState([]);
  
  // Statistics
  const [statistics, setStatistics] = useState({
    total_injuries: 0,
    active_injuries: 0,
    this_week_injuries: 0,
    recovery_rate: 0,
    avg_recovery_time: 0
  });

  // Fetch data functions
  const fetchInjuries = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams({
        q: searchQuery,
        status: filters.status,
        severity: filters.severity,
        injury_type: filters.injury_type,
        team: filters.team,
        league: filters.league,
        confederation: filters.confederation,
        position: filters.position,
        timeframe: filters.timeframe,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: currentPage,
        per_page: itemsPerPage
      });

      const response = await fetch(`${API_BASE_URL}/api/injuries?${queryParams}`);
      if (!response.ok) throw new Error('Failed to fetch injuries');
      
      const data = await response.json();
      setInjuries(data.injuries || []);
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
      const [teamsRes, leaguesRes, injuryTypesRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/teams`),
        fetch(`${API_BASE_URL}/api/leagues`),
        fetch(`${API_BASE_URL}/api/injuries/types`)
      ]);

      if (teamsRes.ok) {
        const teamsData = await teamsRes.json();
        setTeams(teamsData.teams || []);
      }

      if (leaguesRes.ok) {
        const leaguesData = await leaguesRes.json();
        setLeagues(leaguesData.leagues || []);
      }

      if (injuryTypesRes.ok) {
        const typesData = await injuryTypesRes.json();
        setInjuryTypes(typesData.types || []);
      }
    } catch (err) {
      console.error('Error fetching filter data:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchInjuries();
    fetchFilterData();
  }, [searchQuery, filters, sortBy, sortOrder, currentPage]);

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
      injuries: filteredInjuries,
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
    link.download = `injuries_${filters.timeframe}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Filter and sort injuries
  const filteredInjuries = useMemo(() => {
    let filtered = injuries.filter(injury => {
      const matchesSearch = !searchQuery || 
        injury.player?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        injury.team?.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        injury.injury_type?.toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesSearch;
    });

    // Sort injuries
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      if (sortBy === 'date' || sortBy === 'expected_return') {
        aValue = new Date(a[sortBy]);
        bValue = new Date(b[sortBy]);
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [injuries, searchQuery, sortBy, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredInjuries.length / itemsPerPage);
  const paginatedInjuries = filteredInjuries.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Statistics cards
  const statsCards = [
    {
      id: 'total_injuries',
      type: 'injuries',
      icon: AlertTriangle,
      label: 'Total Injuries',
      value: statistics.total_injuries || 0,
      description: 'In selected timeframe',
      change: -2.3,
      changeType: 'positive' // Negative injuries is positive
    },
    {
      id: 'active_injuries',
      type: 'active',
      icon: Activity,
      label: 'Active Injuries',
      value: statistics.active_injuries || 0,
      description: 'Currently injured players',
      progress: ((statistics.active_injuries || 0) / (statistics.total_injuries || 1)) * 100,
      progressColor: 'red'
    },
    {
      id: 'this_week_injuries',
      type: 'recent',
      icon: Clock,
      label: 'This Week',
      value: statistics.this_week_injuries || 0,
      description: 'New injuries this week',
      change: 1.2,
      changeType: statistics.this_week_injuries > 0 ? 'negative' : 'positive'
    },
    {
      id: 'recovery_rate',
      type: 'recovery',
      icon: Heart,
      label: 'Recovery Rate',
      value: `${(statistics.recovery_rate || 0)}%`,
      description: 'Players returning on time',
      progress: statistics.recovery_rate || 0,
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
    { value: 'all', label: 'All Injuries', icon: Activity },
    { value: 'active', label: 'Active', icon: AlertTriangle },
    { value: 'recovered', label: 'Recovered', icon: Heart },
    { value: 'pending', label: 'Pending Assessment', icon: Clock }
  ];

  // Severity options
  const severityOptions = [
    { value: '', label: 'All Severities' },
    { value: 'minor', label: 'Minor (1-2 weeks)' },
    { value: 'moderate', label: 'Moderate (3-8 weeks)' },
    { value: 'major', label: 'Major (2+ months)' },
    { value: 'career_ending', label: 'Career Ending' }
  ];

  // Position options
  const positionOptions = [
    { value: '', label: 'All Positions' },
    { value: 'GK', label: 'Goalkeeper' },
    { value: 'DEF', label: 'Defender' },
    { value: 'MID', label: 'Midfielder' },
    { value: 'FWD', label: 'Forward' }
  ];

  // Get injury severity styling
  const getSeverityStyle = (severity) => {
    switch (severity) {
      case 'minor':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300';
      case 'moderate':
        return 'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/20 dark:text-orange-300';
      case 'major':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300';
      case 'career_ending':
        return 'bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/20 dark:text-purple-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-300';
    }
  };

  // Get status styling
  const getStatusStyle = (status) => {
    switch (status) {
      case 'active':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300';
      case 'recovered':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-300';
    }
  };

  // Format recovery time
  const formatRecoveryTime = (startDate, expectedReturn) => {
    if (!expectedReturn) return 'Unknown';
    
    const start = new Date(startDate);
    const end = new Date(expectedReturn);
    const now = new Date();
    
    const totalDays = Math.ceil((end - start) / (1000 * 60 * 60 * 24));
    const remainingDays = Math.ceil((end - now) / (1000 * 60 * 60 * 24));
    
    if (remainingDays <= 0) return 'Due to return';
    if (remainingDays === 1) return '1 day remaining';
    if (remainingDays < 7) return `${remainingDays} days remaining`;
    if (remainingDays < 30) return `${Math.ceil(remainingDays / 7)} weeks remaining`;
    return `${Math.ceil(remainingDays / 30)} months remaining`;
  };

  // Get injury icon
  const getInjuryIcon = (injuryType) => {
    const iconMap = {
      muscle: Activity,
      ligament: Zap,
      bone: Shield,
      head: AlertTriangle,
      knee: Timer,
      ankle: Users,
      shoulder: BarChart3,
      back: TrendingUp,
      default: AlertTriangle
    };
    
    return iconMap[injuryType?.toLowerCase()] || iconMap.default;
  };

  if (loading && injuries.length === 0) {
    return (
      <div className="p-6">
        <Loading context="injuries" message="Loading injury reports..." />
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
              Error loading injury data
            </h3>
          </div>
          <p className="text-red-600 mt-1 dark:text-red-400">{error}</p>
          <button
            onClick={fetchInjuries}
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
              Player Injuries
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Injury tracking and recovery monitoring across {leagues.length}+ leagues worldwide
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Last updated */}
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>
            
            {/* Refresh button */}
            <button
              onClick={fetchInjuries}
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
            if (statId === 'active_injuries') {
              setFilters(prev => ({ ...prev, status: 'active' }));
            } else if (statId === 'this_week_injuries') {
              setFilters(prev => ({ ...prev, timeframe: 'week' }));
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
                placeholder="Search players, teams, or injury types..."
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

                {/* Severity Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Severity
                  </label>
                  <select
                    value={filters.severity}
                    onChange={(e) => setFilters(prev => ({ ...prev, severity: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    {severityOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Injury Type Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Injury Type
                  </label>
                  <select
                    value={filters.injury_type}
                    onChange={(e) => setFilters(prev => ({ ...prev, injury_type: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  >
                    <option value="">All Injury Types</option>
                    {injuryTypes.map(type => (
                      <option key={type.id} value={type.name}>
                        {type.name}
                      </option>
                    ))}
                  </select>
                </div>

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
              
              {/* Clear Filters */}
              <div className="flex justify-end">
                <button
                  onClick={() => setFilters({
                    status: 'all',
                    severity: '',
                    injury_type: '',
                    team: '',
                    league: '',
                    confederation: '',
                    position: '',
                    timeframe: 'all'
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
            Showing {paginatedInjuries.length} of {filteredInjuries.length} injuries
          </div>
          
          {/* Sort Controls */}
          <div className="flex items-center space-x-4">
            <label className="font-medium">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="date">Injury Date</option>
              <option value="expected_return">Expected Return</option>
              <option value="severity">Severity</option>
              <option value="player">Player Name</option>
              <option value="team">Team</option>
            </select>
            
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Injuries Display */}
        {viewMode === 'cards' ? (
          /* Cards View */
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {paginatedInjuries.map(injury => {
              const InjuryIcon = getInjuryIcon(injury.injury_type);
              
              return (
                <div
                  key={injury.id}
                  className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
                >
                  {/* Injury Header */}
                  <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="bg-red-100 dark:bg-red-900/20 p-2 rounded-lg">
                          <InjuryIcon className="w-5 h-5 text-red-600 dark:text-red-400" />
                        </div>
                        <div>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {injury.injury_type}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {new Date(injury.date).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs font-medium border rounded-full ${getStatusStyle(injury.status)}`}>
                          {injury.status}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium border rounded-full ${getSeverityStyle(injury.severity)}`}>
                          {injury.severity}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Player Info */}
                  <div className="p-4">
                    <button
                      onClick={() => handlePlayerClick(injury.player?.id)}
                      className="flex items-center space-x-3 hover:bg-gray-50 dark:hover:bg-gray-700 p-2 rounded-lg transition-colors w-full"
                    >
                      <img
                        src={injury.player?.photo || '/api/placeholder/40/40'}
                        alt={injury.player?.name}
                        className="w-10 h-10 rounded-full"
                      />
                      <div className="text-left flex-1">
                        <div className="font-medium text-gray-900 dark:text-white">
                          {injury.player?.name}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {injury.player?.position} • Age {injury.player?.age}
                        </div>
                      </div>
                    </button>

                    {/* Team Info */}
                    <button
                      onClick={() => handleTeamClick(injury.team?.id)}
                      className="flex items-center space-x-2 mt-3 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                    >
                      <img
                        src={injury.team?.logo || '/api/placeholder/20/20'}
                        alt={injury.team?.name}
                        className="w-5 h-5 rounded"
                      />
                      <span className="font-medium">{injury.team?.name}</span>
                    </button>

                    {/* Recovery Info */}
                    <div className="mt-4 space-y-2">
                      {injury.expected_return && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Expected Return:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {new Date(injury.expected_return).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Recovery Time:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {formatRecoveryTime(injury.date, injury.expected_return)}
                        </span>
                      </div>

                      {/* Recovery Progress */}
                      {injury.expected_return && injury.status === 'active' && (
                        <div className="mt-3">
                          <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <span>Recovery Progress</span>
                            <span>{injury.recovery_progress || 0}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-600">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${injury.recovery_progress || 0}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Description */}
                    {injury.description && (
                      <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          {injury.description}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
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
                      Team
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Injury
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Severity
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Expected Return
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {paginatedInjuries.map(injury => {
                    const InjuryIcon = getInjuryIcon(injury.injury_type);
                    
                    return (
                      <tr key={injury.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                        {/* Player */}
                        <td className="px-6 py-4">
                          <button
                            onClick={() => handlePlayerClick(injury.player?.id)}
                            className="flex items-center space-x-3 hover:bg-gray-100 dark:hover:bg-gray-600 p-2 rounded-lg transition-colors"
                          >
                            <img
                              src={injury.player?.photo || '/api/placeholder/32/32'}
                              alt={injury.player?.name}
                              className="w-8 h-8 rounded-full"
                            />
                            <div className="text-left">
                              <div className="font-medium text-gray-900 dark:text-white">
                                {injury.player?.name}
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                {injury.player?.position} • {injury.player?.age}y
                              </div>
                            </div>
                          </button>
                        </td>

                        {/* Team */}
                        <td className="px-6 py-4">
                          <button
                            onClick={() => handleTeamClick(injury.team?.id)}
                            className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                          >
                            <img
                              src={injury.team?.logo || '/api/placeholder/20/20'}
                              alt={injury.team?.name}
                              className="w-5 h-5 rounded"
                            />
                            <span className="font-medium">{injury.team?.name}</span>
                          </button>
                        </td>

                        {/* Injury */}
                        <td className="px-6 py-4">
                          <div className="flex items-center space-x-2">
                            <InjuryIcon className="w-4 h-4 text-red-500" />
                            <div>
                              <div className="font-medium text-gray-900 dark:text-white">
                                {injury.injury_type}
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                {new Date(injury.date).toLocaleDateString()}
                              </div>
                            </div>
                          </div>
                        </td>

                        {/* Severity */}
                        <td className="px-6 py-4 text-center">
                          <span className={`px-2 py-1 text-xs font-medium border rounded-full ${getSeverityStyle(injury.severity)}`}>
                            {injury.severity}
                          </span>
                        </td>

                        {/* Status */}
                        <td className="px-6 py-4 text-center">
                          <span className={`px-2 py-1 text-xs font-medium border rounded-full ${getStatusStyle(injury.status)}`}>
                            {injury.status}
                          </span>
                        </td>

                        {/* Expected Return */}
                        <td className="px-6 py-4 text-center">
                          {injury.expected_return ? (
                            <div>
                              <div className="font-medium text-gray-900 dark:text-white">
                                {new Date(injury.expected_return).toLocaleDateString()}
                              </div>
                              <div className="text-xs text-gray-500 dark:text-gray-400">
                                {formatRecoveryTime(injury.date, injury.expected_return)}
                              </div>
                            </div>
                          ) : (
                            <span className="text-gray-400">Unknown</span>
                          )}
                        </td>

                        {/* Actions */}
                        <td className="px-6 py-4 text-right">
                          <button
                            onClick={() => handlePlayerClick(injury.player?.id)}
                            className="p-1 text-blue-600 hover:text-blue-800 transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty State */}
        {filteredInjuries.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <Heart className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No injuries found
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              No injury records match your current filters
            </p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters({
                  status: 'all',
                  severity: '',
                  injury_type: '',
                  team: '',
                  league: '',
                  confederation: '',
                  position: '',
                  timeframe: 'all'
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

export default InjuriesList;