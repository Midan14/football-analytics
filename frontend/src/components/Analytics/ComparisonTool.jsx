import {
    Activity,
    Award,
    BarChart3,
    Download,
    Equal,
    Info,
    RefreshCw,
    Search,
    Target,
    TrendingDown,
    TrendingUp,
    Users,
    X
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
    Bar,
    BarChart,
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

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const ComparisonTool = () => {
  // Estados principales
  const [comparisonType, setComparisonType] = useState('teams');
  const [selectedItems, setSelectedItems] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [chartType, setChartType] = useState('radar');
  const [timeframe, setTimeframe] = useState('season');
  const [metrics, setMetrics] = useState([
    'goals_scored',
    'goals_conceded',
    'xg_for',
    'xg_against',
    'possession_pct',
    'pass_accuracy_pct'
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    league_id: '',
    confederation: '',
    country: '',
    season: '2024-25',
    level: ''
  });

  // Confederaciones desde tu base de datos
  const confederations = ['UEFA', 'CONMEBOL', 'CONCACAF', 'AFC', 'CAF', 'OFC'];
  
  // Métricas disponibles basadas en tu esquema de base de datos
  const availableMetrics = {
    teams: [
      { key: 'goals_scored', label: 'Goals Scored', icon: Target, table: 'team_statistics' },
      { key: 'goals_conceded', label: 'Goals Conceded', icon: Target, table: 'team_statistics' },
      { key: 'xg_for', label: 'Expected Goals (For)', icon: Activity, table: 'team_statistics' },
      { key: 'xg_against', label: 'Expected Goals (Against)', icon: Activity, table: 'team_statistics' },
      { key: 'possession_pct', label: 'Possession %', icon: BarChart3, table: 'team_statistics' },
      { key: 'pass_accuracy_pct', label: 'Pass Accuracy %', icon: TrendingUp, table: 'team_statistics' },
      { key: 'shots_per_game', label: 'Shots per Game', icon: Target, table: 'team_statistics' },
      { key: 'shots_on_target_pct', label: 'Shots on Target %', icon: Target, table: 'team_statistics' },
      { key: 'clean_sheets', label: 'Clean Sheets', icon: Award, table: 'team_statistics' },
      { key: 'wins', label: 'Wins', icon: Award, table: 'team_statistics' },
      { key: 'draws', label: 'Draws', icon: Equal, table: 'team_statistics' },
      { key: 'losses', label: 'Losses', icon: TrendingDown, table: 'team_statistics' }
    ],
    leagues: [
      { key: 'avg_goals_per_match', label: 'Avg Goals per Match', icon: Target, table: 'league_statistics' },
      { key: 'avg_xg_per_match', label: 'Avg xG per Match', icon: Activity, table: 'league_statistics' },
      { key: 'total_matches', label: 'Total Matches', icon: BarChart3, table: 'league_statistics' },
      { key: 'competitive_balance', label: 'Competitive Balance', icon: TrendingUp, table: 'league_statistics' },
      { key: 'avg_attendance', label: 'Avg Attendance', icon: Users, table: 'league_statistics' },
      { key: 'cards_per_match', label: 'Cards per Match', icon: Info, table: 'league_statistics' }
    ],
    players: [
      { key: 'goals', label: 'Goals', icon: Target, table: 'player_statistics' },
      { key: 'assists', label: 'Assists', icon: Users, table: 'player_statistics' },
      { key: 'xg', label: 'Expected Goals', icon: Activity, table: 'player_statistics' },
      { key: 'xa', label: 'Expected Assists', icon: Activity, table: 'player_statistics' },
      { key: 'pass_accuracy_pct', label: 'Pass Accuracy %', icon: TrendingUp, table: 'player_statistics' },
      { key: 'dribbles_completed', label: 'Dribbles Completed', icon: Award, table: 'player_statistics' }
    ]
  };

  // API Functions - Conectadas a tu base de datos real
  const searchItems = async (query, type, filters) => {
    if (query.length < 2) return [];
    
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams({
        q: query,
        ...filters
      }).toString();
      
      const response = await fetch(`${API_BASE_URL}/${type}/search?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.results || [];
    } catch (error) {
      console.error('Error searching items:', error);
      setError('Error searching. Please try again.');
      return [];
    } finally {
      setLoading(false);
    }
  };

  const fetchComparisonData = async (items, selectedMetrics, timeframe) => {
    if (items.length < 2) return null;
    
    setLoading(true);
    setError(null);
    
    try {
      const ids = items.map(item => item.id);
      const response = await fetch(`${API_BASE_URL}/comparison/${comparisonType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ids,
          metrics: selectedMetrics,
          timeframe,
          season: filters.season
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.comparison_data || null;
    } catch (error) {
      console.error('Error fetching comparison data:', error);
      setError('Error loading comparison data. Please try again.');
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Fetch leagues and countries for filters
  const [leagues, setLeagues] = useState([]);
  const [countries, setCountries] = useState([]);

  useEffect(() => {
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
      } catch (error) {
        console.error('Error fetching filter options:', error);
      }
    };

    fetchFilterOptions();
  }, []);

  // Handle search with debouncing
  useEffect(() => {
    if (searchQuery.trim()) {
      const debounceTimer = setTimeout(() => {
        searchItems(searchQuery, comparisonType, filters)
          .then(results => setSearchResults(results));
      }, 300);
      
      return () => clearTimeout(debounceTimer);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery, comparisonType, filters]);

  // Fetch comparison data when items or settings change
  useEffect(() => {
    if (selectedItems.length >= 2) {
      fetchComparisonData(selectedItems, metrics, timeframe)
        .then(data => setComparisonData(data));
    } else {
      setComparisonData(null);
    }
  }, [selectedItems, metrics, timeframe]);

  // Add item to comparison
  const addItemToComparison = (item) => {
    if (selectedItems.length >= 4) {
      setError('Maximum 4 items can be compared at once');
      return;
    }
    
    if (!selectedItems.find(selected => selected.id === item.id)) {
      setSelectedItems([...selectedItems, item]);
      setSearchQuery('');
      setSearchResults([]);
      setError(null);
    }
  };

  // Remove item from comparison
  const removeItemFromComparison = (itemId) => {
    setSelectedItems(selectedItems.filter(item => item.id !== itemId));
  };

  // Prepare data for radar chart
  const radarData = useMemo(() => {
    if (!comparisonData) return [];
    
    return metrics.map(metric => {
      const metricInfo = availableMetrics[comparisonType]?.find(m => m.key === metric);
      const dataPoint = { 
        metric: metricInfo?.label || metric.replace(/_/g, ' ').toUpperCase()
      };
      
      comparisonData.forEach(item => {
        dataPoint[item.name] = parseFloat(item[metric]) || 0;
      });
      
      return dataPoint;
    });
  }, [comparisonData, metrics, comparisonType]);

  // Export functionality
  const exportData = async () => {
    if (!comparisonData) return;
    
    try {
      const exportData = {
        comparison_type: comparisonType,
        timeframe,
        metrics,
        items: selectedItems,
        data: comparisonData,
        generated_at: new Date().toISOString()
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `football-analytics-comparison-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting data:', error);
      setError('Error exporting data. Please try again.');
    }
  };

  // Colors for charts
  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#F97316'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-4xl font-bold text-gray-900 flex items-center gap-3">
              <BarChart3 className="text-blue-600" />
              Football Analytics - Comparison Tool
            </h1>
            <div className="flex gap-3">
              <button
                onClick={exportData}
                disabled={!comparisonData}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-gray-700 transition-colors"
              >
                <Download size={16} />
                Export Data
              </button>
              <button
                onClick={() => {
                  setSelectedItems([]);
                  setSearchQuery('');
                  setSearchResults([]);
                  setComparisonData(null);
                  setError(null);
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <RefreshCw size={16} />
                Reset
              </button>
            </div>
          </div>
          <p className="text-gray-600 text-lg">
            Compare teams, leagues, and players with real-time data from {leagues.length}+ leagues worldwide
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Tipo de Comparación */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Comparison Type
              </label>
              <select
                value={comparisonType}
                onChange={(e) => {
                  setComparisonType(e.target.value);
                  setSelectedItems([]);
                  setComparisonData(null);
                }}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="teams">Teams</option>
                <option value="leagues">Leagues</option>
                <option value="players">Players</option>
              </select>
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

            {/* Timeframe */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Timeframe
              </label>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="season">Full Season 2024-25</option>
                <option value="last10">Last 10 Matches</option>
                <option value="last5">Last 5 Matches</option>
              </select>
            </div>

            {/* Chart Type */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Visualization
              </label>
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="radar">Radar Chart</option>
                <option value="bar">Bar Chart</option>
                <option value="line">Line Chart</option>
                <option value="scatter">Scatter Plot</option>
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Search and Selection Panel */}
          <div className="xl:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Search size={20} />
                Search & Select {comparisonType.charAt(0).toUpperCase() + comparisonType.slice(1)}
              </h2>

              {/* Search Input */}
              <div className="relative mb-6">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                <input
                  type="text"
                  placeholder={`Search ${comparisonType}...`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                {loading && (
                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                    <RefreshCw className="animate-spin text-blue-500" size={16} />
                  </div>
                )}
              </div>

              {/* Search Results */}
              {searchResults.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text-gray-700 mb-2">
                    Search Results ({searchResults.length})
                  </h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {searchResults.map((item) => (
                      <div
                        key={item.id}
                        onClick={() => addItemToComparison(item)}
                        className="p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-blue-50 hover:border-blue-300 transition-colors"
                      >
                        <div className="font-medium text-gray-900">{item.name}</div>
                        <div className="text-sm text-gray-500">
                          {item.league_name || item.country} 
                          {item.confederation && ` • ${item.confederation}`}
                          {item.level && ` • Level ${item.level}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Items */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Selected for Comparison ({selectedItems.length}/4)
                </h3>
                <div className="space-y-2">
                  {selectedItems.map((item, index) => (
                    <div
                      key={item.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: colors[index] }}
                        ></div>
                        <div>
                          <div className="font-medium text-gray-900">{item.name}</div>
                          <div className="text-sm text-gray-500">
                            {item.league_name || item.country}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => removeItemFromComparison(item.id)}
                        className="text-red-500 hover:text-red-700 transition-colors"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ))}
                  
                  {selectedItems.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <Search size={24} className="mx-auto mb-2 opacity-50" />
                      <p>Search and select {comparisonType} to compare</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Metrics Selection */}
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Metrics to Compare
                </h3>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {availableMetrics[comparisonType]?.map((metric) => (
                    <label key={metric.key} className="flex items-center gap-2 hover:bg-gray-50 p-2 rounded">
                      <input
                        type="checkbox"
                        checked={metrics.includes(metric.key)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setMetrics([...metrics, metric.key]);
                          } else {
                            setMetrics(metrics.filter(m => m !== metric.key));
                          }
                        }}
                        className="rounded text-blue-600 focus:ring-blue-500"
                      />
                      <metric.icon size={16} className="text-gray-500" />
                      <span className="text-sm text-gray-700">{metric.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Charts and Analysis Panel */}
          <div className="xl:col-span-2">
            {selectedItems.length >= 2 && comparisonData ? (
              <div className="space-y-8">
                {/* Main Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-6">
                    {comparisonType.charAt(0).toUpperCase() + comparisonType.slice(1)} Performance Comparison
                  </h2>
                  
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      {chartType === 'radar' && (
                        <RadarChart data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="metric" />
                          <PolarRadiusAxis />
                          {selectedItems.map((item, index) => (
                            <Radar
                              key={item.id}
                              name={item.name}
                              dataKey={item.name}
                              stroke={colors[index]}
                              fill={colors[index]}
                              fillOpacity={0.1}
                              strokeWidth={2}
                            />
                          ))}
                          <Legend />
                        </RadarChart>
                      )}
                      
                      {chartType === 'bar' && (
                        <BarChart data={radarData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="metric" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          {selectedItems.map((item, index) => (
                            <Bar
                              key={item.id}
                              dataKey={item.name}
                              fill={colors[index]}
                            />
                          ))}
                        </BarChart>
                      )}
                      
                      {chartType === 'line' && (
                        <LineChart data={radarData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="metric" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          {selectedItems.map((item, index) => (
                            <Line
                              key={item.id}
                              type="monotone"
                              dataKey={item.name}
                              stroke={colors[index]}
                              strokeWidth={2}
                            />
                          ))}
                        </LineChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Statistics Table */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-6">Detailed Statistics</h2>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Metric</th>
                          {selectedItems.map((item, index) => (
                            <th key={item.id} className="text-center py-3 px-4 font-semibold text-gray-700">
                              <div className="flex items-center justify-center gap-2">
                                <div 
                                  className="w-3 h-3 rounded-full" 
                                  style={{ backgroundColor: colors[index] }}
                                ></div>
                                {item.name}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.map((metric) => {
                          const metricInfo = availableMetrics[comparisonType]?.find(m => m.key === metric);
                          return (
                            <tr key={metric} className="border-b border-gray-100 hover:bg-gray-50">
                              <td className="py-3 px-4 font-medium text-gray-900">
                                <div className="flex items-center gap-2">
                                  {metricInfo?.icon && <metricInfo.icon size={16} className="text-gray-500" />}
                                  {metricInfo?.label || metric.replace(/_/g, ' ')}
                                </div>
                              </td>
                              {selectedItems.map((item) => {
                                const value = comparisonData.find(data => data.id === item.id)?.[metric];
                                return (
                                  <td key={item.id} className="py-3 px-4 text-center text-gray-900 font-medium">
                                    {value !== null && value !== undefined ? 
                                      (typeof value === 'number' ? value.toLocaleString() : value) : 
                                      'N/A'
                                    }
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <div className="mb-4">
                  <BarChart3 size={64} className="mx-auto text-gray-300" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Ready to Compare
                </h3>
                <p className="text-gray-600 mb-4">
                  Select at least 2 {comparisonType} from the search panel to start comparing their performance metrics.
                </p>
                <div className="text-sm text-gray-500">
                  💡 Use filters to narrow down your search by league, confederation, or country
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonTool;