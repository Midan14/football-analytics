import {
  Activity,
  Award,
  BarChart3,
  Brain,
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Home,
  Search,
  Settings,
  Star,
  Target,
  TrendingUp,
  Trophy,
  Users
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const Sidebar = ({ isCollapsed, onToggleCollapse }) => {
  // Estados principales
  const [favoriteLeagues, setFavoriteLeagues] = useState([]);
  const [recentMatches, setRecentMatches] = useState([]);
  const [quickStats, setQuickStats] = useState({});
  const [expandedSections, setExpandedSections] = useState({
    analytics: true,
    favorites: true,
    recent: true
  });
  const [loading, setLoading] = useState(false);

  // Hooks de navegación
  const navigate = useNavigate();
  const location = useLocation();

  // Configuración de navegación principal
  const mainNavigation = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      icon: Home,
      description: 'Main overview & insights',
      badge: null
    },
    {
      path: '/predictions',
      label: 'AI Predictions',
      icon: Target,
      description: 'ML-powered match predictions',
      badge: 'NEW'
    },
    {
      path: '/live',
      label: 'Live Matches',
      icon: Activity,
      description: 'Real-time match tracking',
      badge: null,
      isLive: true
    },
    {
      path: '/analytics',
      label: 'Analytics',
      icon: BarChart3,
      description: 'Advanced statistics & comparisons',
      children: [
        { path: '/analytics/comparison', label: 'Comparison Tool', icon: BarChart3 },
        { path: '/analytics/statistics', label: 'Statistics Panel', icon: TrendingUp },
        { path: '/analytics/charts', label: 'Prediction Charts', icon: Brain }
      ]
    },
    {
      path: '/teams',
      label: 'Teams',
      icon: Users,
      description: 'Team profiles & statistics',
      badge: null
    },
    {
      path: '/leagues',
      label: 'Leagues',
      icon: Trophy,
      description: '265+ leagues worldwide',
      badge: '265+'
    }
  ];

  // Quick Actions
  const quickActions = [
    { label: 'Search Teams', path: '/search?type=teams', icon: Search },
    { label: 'Today\'s Matches', path: '/live?date=today', icon: Calendar },
    { label: 'Top Predictions', path: '/predictions?confidence=high', icon: Star },
    { label: 'League Stats', path: '/analytics/statistics?tab=leagues', icon: Award }
  ];

  // Efectos
  useEffect(() => {
    fetchSidebarData();
  }, []);

  // API Functions
  const fetchSidebarData = async () => {
    setLoading(true);
    try {
      // Fetch favorite leagues
      const favoritesResponse = await fetch(`${API_BASE_URL}/user/favorites/leagues`);
      if (favoritesResponse.ok) {
        const favoritesData = await favoritesResponse.json();
        setFavoriteLeagues(favoritesData.leagues || []);
      }

      // Fetch recent matches
      const recentResponse = await fetch(`${API_BASE_URL}/matches/recent?limit=5`);
      if (recentResponse.ok) {
        const recentData = await recentResponse.json();
        setRecentMatches(recentData.matches || []);
      }

      // Fetch quick stats
      const statsResponse = await fetch(`${API_BASE_URL}/stats/quick`);
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setQuickStats(statsData.stats || {});
      }
    } catch (error) {
      console.error('Error fetching sidebar data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handlers
  const handleNavigation = (path) => {
    navigate(path);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const isActiveRoute = (path) => {
    if (path === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(path);
  };

  const addToFavorites = async (leagueId) => {
    try {
      await fetch(`${API_BASE_URL}/user/favorites/leagues/${leagueId}`, {
        method: 'POST'
      });
      fetchSidebarData(); // Refresh data
    } catch (error) {
      console.error('Error adding to favorites:', error);
    }
  };

  return (
    <aside className={`bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-all duration-300 ${
      isCollapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!isCollapsed && (
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">⚽</span>
            </div>
            <div>
              <h2 className="font-bold text-gray-900 dark:text-white text-sm">
                Football Analytics
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                AI-Powered Insights
              </p>
            </div>
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {isCollapsed ? (
            <ChevronRight size={16} className="text-gray-500" />
          ) : (
            <ChevronLeft size={16} className="text-gray-500" />
          )}
        </button>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-2">
          {mainNavigation.map((item) => {
            const IconComponent = item.icon;
            const isActive = isActiveRoute(item.path);
            const hasChildren = item.children && item.children.length > 0;
            const isExpanded = expandedSections[item.path.replace('/', '')];

            return (
              <div key={item.path}>
                <button
                  onClick={() => {
                    if (hasChildren) {
                      toggleSection(item.path.replace('/', ''));
                    } else {
                      handleNavigation(item.path);
                    }
                  }}
                  className={`w-full flex items-center justify-between p-3 rounded-lg text-left transition-colors ${
                    isActive
                      ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                      : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                  }`}
                  title={isCollapsed ? item.label : ''}
                >
                  <div className="flex items-center space-x-3">
                    <div className="relative">
                      <IconComponent size={20} />
                      {item.isLive && (
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
                      )}
                    </div>
                    {!isCollapsed && (
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{item.label}</span>
                          {item.badge && (
                            <span className={`px-2 py-0.5 text-xs rounded-full ${
                              item.badge === 'NEW' 
                                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                            }`}>
                              {item.badge}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                          {item.description}
                        </p>
                      </div>
                    )}
                  </div>
                  {!isCollapsed && hasChildren && (
                    <ChevronDown 
                      size={16} 
                      className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    />
                  )}
                </button>

                {/* Submenu */}
                {!isCollapsed && hasChildren && isExpanded && (
                  <div className="ml-6 mt-2 space-y-1">
                    {item.children.map((child) => {
                      const ChildIcon = child.icon;
                      const isChildActive = isActiveRoute(child.path);
                      
                      return (
                        <button
                          key={child.path}
                          onClick={() => handleNavigation(child.path)}
                          className={`w-full flex items-center space-x-3 p-2 rounded-md text-sm transition-colors ${
                            isChildActive
                              ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                              : 'text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800'
                          }`}
                        >
                          <ChildIcon size={16} />
                          <span>{child.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Quick Actions */}
        {!isCollapsed && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
              Quick Actions
            </h3>
            <div className="space-y-1">
              {quickActions.map((action, index) => {
                const IconComponent = action.icon;
                return (
                  <button
                    key={index}
                    onClick={() => handleNavigation(action.path)}
                    className="w-full flex items-center space-x-3 p-2 rounded-md text-sm text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800 transition-colors"
                  >
                    <IconComponent size={16} />
                    <span>{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Quick Stats */}
        {!isCollapsed && Object.keys(quickStats).length > 0 && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
              Quick Stats
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Active Leagues</span>
                <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                  {quickStats.active_leagues || 265}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Today's Matches</span>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  {quickStats.todays_matches || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Predictions Made</span>
                <span className="text-sm font-semibold text-purple-600 dark:text-purple-400">
                  {quickStats.predictions_made || 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Model Accuracy</span>
                <span className="text-sm font-semibold text-orange-600 dark:text-orange-400">
                  {quickStats.model_accuracy ? `${(quickStats.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Favorite Leagues */}
        {!isCollapsed && favoriteLeagues.length > 0 && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={() => toggleSection('favorites')}
              className="w-full flex items-center justify-between mb-3"
            >
              <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Favorite Leagues
              </h3>
              <ChevronDown 
                size={14} 
                className={`transition-transform ${expandedSections.favorites ? 'rotate-180' : ''}`}
              />
            </button>
            
            {expandedSections.favorites && (
              <div className="space-y-2">
                {favoriteLeagues.slice(0, 5).map((league) => (
                  <button
                    key={league.id}
                    onClick={() => handleNavigation(`/leagues/${league.id}`)}
                    className="w-full flex items-center justify-between p-2 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-gradient-to-br from-blue-400 to-green-400 rounded-sm flex items-center justify-center">
                        <Trophy size={10} className="text-white" />
                      </div>
                      <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                        {league.name}
                      </span>
                    </div>
                    <Star size={12} className="text-yellow-500" />
                  </button>
                ))}
                
                {favoriteLeagues.length > 5 && (
                  <button
                    onClick={() => handleNavigation('/leagues?filter=favorites')}
                    className="w-full text-left p-2 text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    View all {favoriteLeagues.length} favorites →
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* Recent Matches */}
        {!isCollapsed && recentMatches.length > 0 && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={() => toggleSection('recent')}
              className="w-full flex items-center justify-between mb-3"
            >
              <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Recent Matches
              </h3>
              <ChevronDown 
                size={14} 
                className={`transition-transform ${expandedSections.recent ? 'rotate-180' : ''}`}
              />
            </button>
            
            {expandedSections.recent && (
              <div className="space-y-2">
                {recentMatches.slice(0, 3).map((match) => (
                  <button
                    key={match.id}
                    onClick={() => handleNavigation(`/matches/${match.id}`)}
                    className="w-full p-2 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex-1 text-left">
                        <div className="font-medium text-gray-900 dark:text-white truncate">
                          {match.home_team} vs {match.away_team}
                        </div>
                        <div className="text-gray-500 dark:text-gray-400">
                          {match.league_name}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-gray-900 dark:text-white">
                          {match.score || 'TBD'}
                        </div>
                        <div className="text-gray-500 dark:text-gray-400">
                          {new Date(match.date).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
                
                <button
                  onClick={() => handleNavigation('/matches')}
                  className="w-full text-left p-2 text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                >
                  View all matches →
                </button>
              </div>
            )}
          </div>
        )}

        {/* Bottom Actions */}
        {!isCollapsed && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={() => handleNavigation('/settings')}
              className="w-full flex items-center space-x-3 p-2 rounded-md text-sm text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800 transition-colors"
            >
              <Settings size={16} />
              <span>Settings</span>
            </button>
          </div>
        )}
      </nav>

      {/* Collapsed Mode Quick Access */}
      {isCollapsed && (
        <div className="p-2 border-t border-gray-200 dark:border-gray-700">
          <div className="space-y-2">
            <button
              onClick={() => handleNavigation('/predictions')}
              className="w-full p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="AI Predictions"
            >
              <Target size={20} className="text-green-600 mx-auto" />
            </button>
            <button
              onClick={() => handleNavigation('/live')}
              className="w-full p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors relative"
              title="Live Matches"
            >
              <Activity size={20} className="text-red-600 mx-auto" />
              <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-ping"></div>
            </button>
            <button
              onClick={() => handleNavigation('/settings')}
              className="w-full p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="Settings"
            >
              <Settings size={20} className="text-gray-500 mx-auto" />
            </button>
          </div>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;