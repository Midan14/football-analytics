import {
    Activity,
    ArrowDownRight,
    ArrowUpRight,
    Award,
    BarChart3,
    Clock,
    Download,
    Hash,
    Minus,
    Percent,
    RefreshCw,
    Search,
    Target,
    TrendingDown,
    TrendingUp,
    Trophy,
    Users,
    Zap
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const StatisticsPanel = () => {
  // Estados principales
  const [activeTab, setActiveTab] = useState('teams'); // teams, players, leagues
  const [viewMode, setViewMode] = useState('table'); // table, charts, rankings
  const [statisticsData, setStatisticsData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Estados de filtros
  const [filters, setFilters] = useState({
    league_id: '',
    confederation: '',
    country: '',
    season: '2024-25',
    position: '', // Solo para players
    team_id: '', // Solo para players
    level: '', // Solo para leagues
    timeframe: 'season' // season, last10, last5
  });

  // Estados de búsqueda y paginación
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('');
  const [sortOrder, setSortOrder] = useState('desc');
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 50,
    total: 0
  });

  // Datos de filtros
  const [leagues, setLeagues] = useState([]);
  const [countries, setCountries] = useState([]);
  const [teams, setTeams] = useState([]);
  const confederations = ['UEFA', 'CONMEBOL', 'CONCACAF', 'AFC', 'CAF', 'OFC'];
  const positions = ['GK', 'DEF', 'MID', 'FWD'];

  // Configuración de tabs
  const tabs = [
    { key: 'teams', label: 'Team Statistics', icon: Users },
    { key: 'players', label: 'Player Statistics', icon: Award },
    { key: 'leagues', label: 'League Statistics', icon: Trophy }
  ];

  // Configuración de métricas por tab
  const metrics = {
    teams: [
      { key: 'goals_scored', label: 'Goals Scored', icon: Target, sortable: true },
      { key: 'goals_conceded', label: 'Goals Conceded', icon: Target, sortable: true },
      { key: 'xg_for', label: 'xG For', icon: Activity, sortable: true },
      { key: 'xg_against', label: 'xG Against', icon: Activity, sortable: true },
      { key: 'possession_pct', label: 'Possession %', icon: Percent, sortable: true },
      { key: 'pass_accuracy_pct', label: 'Pass Accuracy %', icon: TrendingUp, sortable: true },
      { key: 'shots_per_game', label: 'Shots/Game', icon: Target, sortable: true },
      { key: 'clean_sheets', label: 'Clean Sheets', icon: Award, sortable: true },
      { key: 'wins', label: 'Wins', icon: ArrowUpRight, sortable: true },
      { key: 'draws', label: 'Draws', icon: Minus, sortable: true },
      { key: 'losses', label: 'Losses', icon: ArrowDownRight, sortable: true }
    ],
    players: [
      { key: 'goals', label: 'Goals', icon: Target, sortable: true },
      { key: 'assists', label: 'Assists', icon: Users, sortable: true },
      { key: 'xg', label: 'Expected Goals', icon: Activity, sortable: true },
      { key: 'xa', label: 'Expected Assists', icon: Activity, sortable: true },
      { key: 'pass_accuracy_pct', label: 'Pass Accuracy %', icon: TrendingUp, sortable: true },
      { key: 'dribbles_completed', label: 'Dribbles', icon: Zap, sortable: true },
      { key: 'minutes_played', label: 'Minutes', icon: Clock, sortable: true },
      { key: 'yellow_cards', label: 'Yellow Cards', icon: Hash, sortable: true },
      { key: 'red_cards', label: 'Red Cards', icon: Hash, sortable: true }
    ],
    leagues: [
      { key: 'avg_goals_per_match', label: 'Avg Goals/Match', icon: Target, sortable: true },
      { key: 'avg_xg_per_match', label: 'Avg xG/Match', icon: Activity, sortable: true },
      { key: 'total_matches', label: 'Total Matches', icon: Hash, sortable: true },
      { key: 'competitive_balance', label: 'Competitive Balance', icon: BarChart3, sortable: true },
      { key: 'avg_attendance', label: 'Avg Attendance', icon: Users, sortable: true },
      { key: 'cards_per_match', label: 'Cards/Match', icon: Hash, sortable: true }
    ]
  };

  // API Functions
  const fetchStatistics = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams({
        ...filters,
        search: searchQuery,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: pagination.page,
        limit: pagination.limit
      }).toString();
      
      const response = await fetch(`${API_BASE_URL}/statistics/${activeTab}?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      setStatisticsData(data.results || []);
      setPagination(prev => ({
        ...prev,
        total: data.pagination?.total || 0
      }));
      
    } catch (error) {
      console.error('Error fetching statistics:', error);
      setError('Error loading statistics. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchFilterOptions = async () => {
    try {
      // Fetch leagues
      const leaguesResponse = await fetch(`${API_BASE_URL}/leagues?is_active=true`);
      if (leaguesResponse.ok) {
        const leaguesData = await leaguesResponse.json();
        setLeagues(leaguesData.leagues || []);
      }

      // Fetch countries
      const countriesResponse = await fetch(`${API_BASE_URL}/countries`);
      if (countriesResponse.ok) {
        const countriesData = await countriesResponse.json();
        setCountries(countriesData.countries || []);
      }

      // Fetch teams (for player filtering)
      if (activeTab === 'players') {
        const teamsResponse = await fetch(`${API_BASE_URL}/teams/search?limit=100`);
        if (teamsResponse.ok) {
          const teamsData = await teamsResponse.json();
          setTeams(teamsData.results || []);
        }
      }
    } catch (error) {
      console.error('Error fetching filter options:', error);
    }
  };

  // Effects
  useEffect(() => {
    fetchFilterOptions();
  }, [activeTab]);

  useEffect(() => {
    fetchStatistics();
  }, [activeTab, filters, searchQuery, sortBy, sortOrder, pagination.page]);

  // Handlers
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setStatisticsData([]);
    setSearchQuery('');
    setSortBy('');
    setPagination(prev => ({ ...prev, page: 1 }));
    setFilters({
      league_id: '',
      confederation: '',
      country: '',
      season: '2024-25',
      position: '',
      team_id: '',
      level: '',
      timeframe: 'season'
    });
  };

  const handleSort = (metric) => {
    if (!metric.sortable) return;
    
    if (sortBy === metric.key) {
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(metric.key);
      setSortOrder('desc');
    }
    setPagination(prev => ({ ...prev, page: 1 }));
  };

  const handlePageChange = (newPage) => {
    setPagination(prev => ({ ...prev, page: newPage }));
  };

  // Export functionality
  const exportData = async () => {
    try {
      const exportData = {
        type: activeTab,
        data: statisticsData,
        filters,
        search: searchQuery,
        sort: { by: sortBy, order: sortOrder },
        generated_at: new Date().toISOString(),
        total_records: pagination.total
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `football-${activeTab}-statistics-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting data:', error);
      setError('Error exporting data. Please try again.');
    }
  };

  // Prepare chart data
  const chartData = useMemo(() => {
    if (statisticsData.length === 0) return [];
    
    return statisticsData.slice(0, 10).map(item => {
      const baseData = { name: item.name || `${item.first_name} ${item.last_name}` || item.league_name };
      
      metrics[activeTab].forEach(metric => {
        baseData[metric.key] = parseFloat(item[metric.key]) || 0;
      });
      
      return baseData;
    });
  }, [statisticsData, activeTab]);

  // Get ranking change indicator
  const getRankingChange = (item) => {
    if (!item.rank_change) return null;
    
    if (item.rank_change > 0) {
      return <ArrowUpRight className="text-green-600" size={16} />;
    } else if (item.rank_change < 0) {
      return <ArrowDownRight className="text-red-600" size={16} />;
    }
    return <Minus className="text-gray-400" size={16} />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-4xl font-bold text-gray-900 flex items-center gap-3">
              <BarChart3 className="text-blue-600" />
              Football Statistics Panel
            </h1>
            <div className="flex gap-3">
              <button
                onClick={exportData}
                disabled={statisticsData.length === 0}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-gray-700 transition-colors"
              >
                <Download size={16} />
                Export Data
              </button>
              <button
                onClick={fetchStatistics}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <RefreshCw size={16} />
                Refresh
              </button>
            </div>
          </div>
          <p className="text-gray-600 text-lg">
            Real-time statistics from {leagues.length}+ leagues worldwide
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex flex-wrap gap-2 mb-6">
            {tabs.map(tab => {
              const IconComponent = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => handleTabChange(tab.key)}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                    activeTab === tab.key
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <IconComponent size={16} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* View Mode Toggle */}
          <div className="flex gap-2">
            {['table', 'charts', 'rankings'].map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 text-sm rounded ${
                  viewMode === mode
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-6 gap-6">
            {/* Search */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Search
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                <input
                  type="text"
                  placeholder={`Search ${activeTab}...`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* League Filter */}
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

            {/* Confederation Filter */}
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

            {/* Position Filter (Only for Players) */}
            {activeTab === 'players' && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Position
                </label>
                <select
                  value={filters.position}
                  onChange={(e) => setFilters({...filters, position: e.target.value})}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Positions</option>
                  {positions.map(pos => (
                    <option key={pos} value={pos}>{pos}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Team Filter (Only for Players) */}
            {activeTab === 'players' && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Team
                </label>
                <select
                  value={filters.team_id}
                  onChange={(e) => setFilters({...filters, team_id: e.target.value})}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Teams</option>
                  {teams.map(team => (
                    <option key={team.id} value={team.id}>
                      {team.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Level Filter (Only for Leagues) */}
            {activeTab === 'leagues' && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Division Level
                </label>
                <select
                  value={filters.level}
                  onChange={(e) => setFilters({...filters, level: e.target.value})}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Levels</option>
                  <option value="1">1st Division</option>
                  <option value="2">2nd Division</option>
                  <option value="3">3rd Division</option>
                </select>
              </div>
            )}

            {/* Timeframe Filter */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Timeframe
              </label>
              <select
                value={filters.timeframe}
                onChange={(e) => setFilters({...filters, timeframe: e.target.value})}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="season">Full Season</option>
                <option value="last10">Last 10 Matches</option>
                <option value="last5">Last 5 Matches</option>
              </select>
            </div>
          </div>
        </div>

        {/* Main Content */}
        {loading ? (
          <div className="bg-white rounded-xl shadow-lg p-8 text-center">
            <RefreshCw className="animate-spin mx-auto mb-4 text-blue-500" size={32} />
            <p className="text-gray-600">Loading statistics...</p>
          </div>
        ) : (
          <>
            {/* Table View */}
            {viewMode === 'table' && (
              <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                <div className="p-6 border-b border-gray-200">
                  <h2 className="text-xl font-bold text-gray-900">
                    {tabs.find(t => t.key === activeTab)?.label}
                    <span className="ml-2 text-gray-500 font-normal">
                      ({pagination.total.toLocaleString()} results)
                    </span>
                  </h2>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Rank</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Name</th>
                        {activeTab === 'teams' && (
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">League</th>
                        )}
                        {activeTab === 'players' && (
                          <>
                            <th className="text-left py-3 px-4 font-semibold text-gray-700">Team</th>
                            <th className="text-left py-3 px-4 font-semibold text-gray-700">Position</th>
                          </>
                        )}
                        {activeTab === 'leagues' && (
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Country</th>
                        )}
                        
                        {metrics[activeTab].slice(0, 6).map(metric => {
                          const IconComponent = metric.icon;
                          return (
                            <th
                              key={metric.key}
                              onClick={() => handleSort(metric)}
                              className={`text-center py-3 px-4 font-semibold text-gray-700 ${
                                metric.sortable ? 'cursor-pointer hover:bg-gray-100' : ''
                              }`}
                            >
                              <div className="flex items-center justify-center gap-1">
                                <IconComponent size={14} />
                                {metric.label}
                                {sortBy === metric.key && (
                                  sortOrder === 'desc' 
                                    ? <TrendingDown size={12} />
                                    : <TrendingUp size={12} />
                                )}
                              </div>
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {statisticsData.map((item, index) => (
                        <tr key={item.id} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {(pagination.page - 1) * pagination.limit + index + 1}
                              </span>
                              {getRankingChange(item)}
                            </div>
                          </td>
                          <td className="py-3 px-4 font-medium text-gray-900">
                            {item.name || `${item.first_name} ${item.last_name}` || item.league_name}
                          </td>
                          {activeTab === 'teams' && (
                            <td className="py-3 px-4 text-gray-600">{item.league_name}</td>
                          )}
                          {activeTab === 'players' && (
                            <>
                              <td className="py-3 px-4 text-gray-600">{item.team_name}</td>
                              <td className="py-3 px-4 text-gray-600">{item.position}</td>
                            </>
                          )}
                          {activeTab === 'leagues' && (
                            <td className="py-3 px-4 text-gray-600">{item.country}</td>
                          )}
                          
                          {metrics[activeTab].slice(0, 6).map(metric => (
                            <td key={metric.key} className="py-3 px-4 text-center font-medium">
                              {item[metric.key] !== null && item[metric.key] !== undefined
                                ? typeof item[metric.key] === 'number'
                                  ? item[metric.key].toLocaleString()
                                  : item[metric.key]
                                : 'N/A'
                              }
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="p-6 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      Showing {(pagination.page - 1) * pagination.limit + 1} to{' '}
                      {Math.min(pagination.page * pagination.limit, pagination.total)} of{' '}
                      {pagination.total.toLocaleString()} results
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handlePageChange(pagination.page - 1)}
                        disabled={pagination.page === 1}
                        className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                      >
                        Previous
                      </button>
                      <span className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded">
                        Page {pagination.page}
                      </span>
                      <button
                        onClick={() => handlePageChange(pagination.page + 1)}
                        disabled={pagination.page * pagination.limit >= pagination.total}
                        className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Charts View */}
            {viewMode === 'charts' && chartData.length > 0 && (
              <div className="space-y-8">
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-6">Top 10 Performance Chart</h2>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {metrics[activeTab].slice(0, 3).map((metric, index) => (
                          <Bar
                            key={metric.key}
                            dataKey={metric.key}
                            name={metric.label}
                            fill={['#3B82F6', '#10B981', '#F59E0B'][index]}
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* Rankings View */}
            {viewMode === 'rankings' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {metrics[activeTab].slice(0, 4).map(metric => {
                  const topItems = statisticsData.slice(0, 5);
                  const IconComponent = metric.icon;
                  
                  return (
                    <div key={metric.key} className="bg-white rounded-xl shadow-lg p-6">
                      <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                        <IconComponent size={20} />
                        Top 5 - {metric.label}
                      </h3>
                      <div className="space-y-3">
                        {topItems.map((item, index) => (
                          <div key={item.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center font-bold text-sm">
                                {index + 1}
                              </div>
                              <div>
                                <div className="font-medium text-gray-900">
                                  {item.name || `${item.first_name} ${item.last_name}` || item.league_name}
                                </div>
                                <div className="text-sm text-gray-500">
                                  {item.league_name || item.team_name || item.country}
                                </div>
                              </div>
                            </div>
                            <div className="font-bold text-lg text-blue-600">
                              {item[metric.key] !== null && item[metric.key] !== undefined
                                ? typeof item[metric.key] === 'number'
                                  ? item[metric.key].toLocaleString()
                                  : item[metric.key]
                                : 'N/A'
                              }
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default StatisticsPanel;