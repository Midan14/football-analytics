import {
  Activity,
  BarChart3,
  Bell,
  ChevronDown,
  Globe,
  Home,
  LogOut,
  Menu,
  Moon,
  Search,
  Settings,
  Sun,
  Target,
  Trophy,
  User,
  Users,
  X
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const Header = () => {
  // Estados principales
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');

  // Referencias para dropdowns
  const searchRef = useRef(null);
  const profileRef = useRef(null);
  const notificationsRef = useRef(null);

  // Hooks de navegación
  const navigate = useNavigate();
  const location = useLocation();

  // Configuración de navegación
  const navigationItems = [
    { 
      path: '/dashboard', 
      label: 'Dashboard', 
      icon: Home,
      description: 'Main overview'
    },
    { 
      path: '/predictions', 
      label: 'Predictions', 
      icon: Target,
      description: 'AI-powered match predictions'
    },
    { 
      path: '/live', 
      label: 'Live Matches', 
      icon: Activity,
      description: 'Real-time match tracking'
    },
    { 
      path: '/analytics', 
      label: 'Analytics', 
      icon: BarChart3,
      description: 'Advanced statistics & comparisons',
      children: [
        { path: '/analytics/comparison', label: 'Comparison Tool' },
        { path: '/analytics/statistics', label: 'Statistics Panel' },
        { path: '/analytics/charts', label: 'Prediction Charts' }
      ]
    },
    { 
      path: '/teams', 
      label: 'Teams', 
      icon: Users,
      description: 'Team profiles & stats'
    },
    { 
      path: '/leagues', 
      label: 'Leagues', 
      icon: Trophy,
      description: '265+ leagues worldwide'
    }
  ];

  // Usuario mock - en producción vendría de context/auth
  const user = {
    name: 'Football Analyst',
    email: 'analyst@football-analytics.com',
    avatar: null,
    role: 'Premium User',
    preferences: {
      favorite_leagues: ['Premier League', 'La Liga', 'Champions League'],
      timezone: 'America/Bogota'
    }
  };

  // Efectos
  useEffect(() => {
    // Aplicar tema
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    // Cerrar dropdowns al hacer click fuera
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setIsSearchOpen(false);
      }
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setIsProfileOpen(false);
      }
      if (notificationsRef.current && !notificationsRef.current.contains(event.target)) {
        setIsNotificationsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    // Fetch notifications
    fetchNotifications();
  }, []);

  useEffect(() => {
    // Búsqueda con debounce
    if (searchQuery.trim().length >= 2) {
      const debounceTimer = setTimeout(() => {
        performSearch(searchQuery);
      }, 300);
      
      return () => clearTimeout(debounceTimer);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  // API Functions
  const performSearch = async (query) => {
    setSearchLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/search/global?q=${encodeURIComponent(query)}`);
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
      }
    } catch (error) {
      console.error('Error performing search:', error);
    } finally {
      setSearchLoading(false);
    }
  };

  const fetchNotifications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/notifications?limit=10`);
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unread_count || 0);
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };

  const markNotificationAsRead = async (notificationId) => {
    try {
      await fetch(`${API_BASE_URL}/notifications/${notificationId}/read`, {
        method: 'POST'
      });
      
      setNotifications(notifications.map(notif => 
        notif.id === notificationId ? { ...notif, is_read: true } : notif
      ));
      
      setUnreadCount(Math.max(0, unreadCount - 1));
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  };

  // Handlers
  const handleNavigation = (path) => {
    navigate(path);
    setIsMenuOpen(false);
  };

  const handleSearchSelect = (result) => {
    if (result.type === 'team') {
      navigate(`/teams/${result.id}`);
    } else if (result.type === 'league') {
      navigate(`/leagues/${result.id}`);
    } else if (result.type === 'player') {
      navigate(`/players/${result.id}`);
    } else if (result.type === 'match') {
      navigate(`/matches/${result.id}`);
    }
    
    setSearchQuery('');
    setSearchResults([]);
    setIsSearchOpen(false);
  };

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    navigate('/login');
  };

  // Check if route is active
  const isActiveRoute = (path) => {
    if (path === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <header className="bg-white dark:bg-gray-900 shadow-lg border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-green-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">⚽</span>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  Football Analytics
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  AI-Powered Insights
                </p>
              </div>
            </button>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navigationItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = isActiveRoute(item.path);
              
              return (
                <div key={item.path} className="relative group">
                  <button
                    onClick={() => handleNavigation(item.path)}
                    className={`px-4 py-2 rounded-lg flex items-center space-x-2 text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                        : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                    }`}
                  >
                    <IconComponent size={16} />
                    <span>{item.label}</span>
                    {item.children && <ChevronDown size={14} />}
                  </button>
                  
                  {/* Submenu */}
                  {item.children && (
                    <div className="absolute top-full left-0 mt-1 w-56 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                      <div className="p-2">
                        {item.children.map((child) => (
                          <button
                            key={child.path}
                            onClick={() => handleNavigation(child.path)}
                            className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
                          >
                            {child.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </nav>

          {/* Search, Notifications, Profile */}
          <div className="flex items-center space-x-4">
            {/* Global Search */}
            <div className="relative" ref={searchRef}>
              <button
                onClick={() => setIsSearchOpen(!isSearchOpen)}
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              >
                <Search size={20} />
              </button>

              {/* Search Dropdown */}
              {isSearchOpen && (
                <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                  <div className="p-4">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                      <input
                        type="text"
                        placeholder="Search teams, leagues, players..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        autoFocus
                      />
                    </div>

                    {/* Search Results */}
                    {searchLoading && (
                      <div className="mt-3 text-center py-4">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
                      </div>
                    )}

                    {searchResults.length > 0 && (
                      <div className="mt-3 max-h-64 overflow-y-auto">
                        {searchResults.map((result) => (
                          <button
                            key={`${result.type}-${result.id}`}
                            onClick={() => handleSearchSelect(result)}
                            className="w-full text-left p-3 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                          >
                            <div className="flex items-center space-x-3">
                              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                                {result.type === 'team' && <Users size={14} />}
                                {result.type === 'league' && <Trophy size={14} />}
                                {result.type === 'player' && <User size={14} />}
                                {result.type === 'match' && <Activity size={14} />}
                              </div>
                              <div className="flex-1">
                                <div className="font-medium text-gray-900 dark:text-white">
                                  {result.name}
                                </div>
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                  {result.subtitle}
                                </div>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}

                    {searchQuery.length >= 2 && searchResults.length === 0 && !searchLoading && (
                      <div className="mt-3 text-center py-4 text-gray-500 dark:text-gray-400">
                        No results found
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Notifications */}
            <div className="relative" ref={notificationsRef}>
              <button
                onClick={() => setIsNotificationsOpen(!isNotificationsOpen)}
                className="relative p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              >
                <Bell size={20} />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                )}
              </button>

              {/* Notifications Dropdown */}
              {isNotificationsOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                  <div className="p-4">
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                      Notifications
                    </h3>
                    
                    {notifications.length > 0 ? (
                      <div className="space-y-3 max-h-64 overflow-y-auto">
                        {notifications.map((notification) => (
                          <div
                            key={notification.id}
                            onClick={() => markNotificationAsRead(notification.id)}
                            className={`p-3 rounded-lg cursor-pointer transition-colors ${
                              notification.is_read
                                ? 'bg-gray-50 dark:bg-gray-700'
                                : 'bg-blue-50 dark:bg-blue-900'
                            }`}
                          >
                            <div className="flex items-start space-x-3">
                              <div className="flex-shrink-0">
                                {notification.type === 'prediction' && <Target size={16} />}
                                {notification.type === 'match' && <Activity size={16} />}
                                {notification.type === 'league' && <Trophy size={16} />}
                                {notification.type === 'general' && <Bell size={16} />}
                              </div>
                              <div className="flex-1">
                                <p className="text-sm font-medium text-gray-900 dark:text-white">
                                  {notification.title}
                                </p>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                                  {new Date(notification.created_at).toLocaleString()}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500 dark:text-gray-400">
                        No notifications
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>

            {/* Profile */}
            <div className="relative" ref={profileRef}>
              <button
                onClick={() => setIsProfileOpen(!isProfileOpen)}
                className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-green-500 rounded-full flex items-center justify-center">
                  <User className="text-white" size={16} />
                </div>
                <div className="hidden sm:block text-left">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {user.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {user.role}
                  </p>
                </div>
                <ChevronDown size={14} className="text-gray-400" />
              </button>

              {/* Profile Dropdown */}
              {isProfileOpen && (
                <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                  <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-green-500 rounded-full flex items-center justify-center">
                        <User className="text-white" size={20} />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          {user.name}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {user.email}
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-2">
                    <button
                      onClick={() => navigate('/profile')}
                      className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors flex items-center space-x-2"
                    >
                      <User size={16} />
                      <span>Profile</span>
                    </button>
                    <button
                      onClick={() => navigate('/settings')}
                      className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors flex items-center space-x-2"
                    >
                      <Settings size={16} />
                      <span>Settings</span>
                    </button>
                    <button
                      onClick={() => navigate('/help')}
                      className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors flex items-center space-x-2"
                    >
                      <Globe size={16} />
                      <span>Help Center</span>
                    </button>
                    <hr className="my-2 border-gray-200 dark:border-gray-700" />
                    <button
                      onClick={handleLogout}
                      className="w-full text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900 rounded-md transition-colors flex items-center space-x-2"
                    >
                      <LogOut size={16} />
                      <span>Sign Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Mobile Menu Toggle */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="lg:hidden p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            >
              {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="lg:hidden border-t border-gray-200 dark:border-gray-700">
            <nav className="py-4 space-y-2">
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                const isActive = isActiveRoute(item.path);
                
                return (
                  <button
                    key={item.path}
                    onClick={() => handleNavigation(item.path)}
                    className={`w-full text-left px-4 py-3 rounded-lg flex items-center space-x-3 transition-colors ${
                      isActive
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                        : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                    }`}
                  >
                    <IconComponent size={20} />
                    <div>
                      <div className="font-medium">{item.label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {item.description}
                      </div>
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;