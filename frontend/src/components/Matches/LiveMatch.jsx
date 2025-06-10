import {
    Activity,
    ArrowLeft,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Brain,
    Calendar,
    CheckCircle,
    Clock,
    Crosshair,
    Flag,
    Hash,
    MapPin,
    MessageSquare,
    Pause,
    Play,
    Share2,
    Target,
    Users,
    Volume2,
    VolumeX,
    Zap
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

// Import components
import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const LiveMatch = () => {
  // URL Parameters
  const { matchId } = useParams();
  const navigate = useNavigate();

  // Estados principales
  const [match, setMatch] = useState(null);
  const [matchStats, setMatchStats] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [lineups, setLineups] = useState({ home: [], away: [] });
  const [predictions, setPredictions] = useState(null);
  const [commentary, setCommentary] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Estados de configuración
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(10000); // 10 segundos
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [selectedTab, setSelectedTab] = useState('overview'); // overview, events, lineups, stats, predictions
  const [isFavorite, setIsFavorite] = useState(false);

  // Referencias
  const refreshIntervalRef = useRef(null);
  const audioRef = useRef(null);

  // Efectos
  useEffect(() => {
    if (matchId) {
      fetchMatchData();
      checkIfFavorite();
    }
    
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [matchId]);

  useEffect(() => {
    if (autoRefresh && match?.status === 'live') {
      refreshIntervalRef.current = setInterval(() => {
        fetchLiveData();
      }, refreshInterval);
    } else {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [autoRefresh, match?.status, refreshInterval]);

  // API Functions
  const fetchMatchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [
        matchResponse,
        statsResponse,
        eventsResponse,
        lineupsResponse,
        predictionsResponse,
        commentaryResponse
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/matches/${matchId}`),
        fetch(`${API_BASE_URL}/matches/${matchId}/statistics`),
        fetch(`${API_BASE_URL}/matches/${matchId}/events`),
        fetch(`${API_BASE_URL}/matches/${matchId}/lineups`),
        fetch(`${API_BASE_URL}/matches/${matchId}/predictions`),
        fetch(`${API_BASE_URL}/matches/${matchId}/commentary`)
      ]);

      if (!matchResponse.ok) {
        throw new Error('Match not found');
      }

      const matchData = await matchResponse.json();
      const statsData = statsResponse.ok ? await statsResponse.json() : null;
      const eventsData = eventsResponse.ok ? await eventsResponse.json() : null;
      const lineupsData = lineupsResponse.ok ? await lineupsResponse.json() : null;
      const predictionsData = predictionsResponse.ok ? await predictionsResponse.json() : null;
      const commentaryData = commentaryResponse.ok ? await commentaryResponse.json() : null;

      setMatch(matchData.match);
      setMatchStats(statsData?.statistics);
      setLiveEvents(eventsData?.events || []);
      setLineups(lineupsData?.lineups || { home: [], away: [] });
      setPredictions(predictionsData?.predictions);
      setCommentary(commentaryData?.commentary || []);

    } catch (error) {
      console.error('Error fetching match data:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchLiveData = async () => {
    if (!match || match.status !== 'live') return;

    try {
      const [statsResponse, eventsResponse, commentaryResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/matches/${matchId}/statistics`),
        fetch(`${API_BASE_URL}/matches/${matchId}/events`),
        fetch(`${API_BASE_URL}/matches/${matchId}/commentary?latest=5`)
      ]);

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setMatchStats(statsData.statistics);
      }

      if (eventsResponse.ok) {
        const eventsData = await eventsResponse.json();
        const newEvents = eventsData.events || [];
        
        // Check for new events and play sound
        if (newEvents.length > liveEvents.length && soundEnabled) {
          playNotificationSound();
        }
        
        setLiveEvents(newEvents);
      }

      if (commentaryResponse.ok) {
        const commentaryData = await commentaryResponse.json();
        setCommentary(prev => [...(commentaryData.commentary || []), ...prev].slice(0, 50));
      }

    } catch (error) {
      console.error('Error fetching live data:', error);
    }
  };

  const checkIfFavorite = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/favorites/matches/${matchId}`);
      setIsFavorite(response.ok);
    } catch (error) {
      console.error('Error checking favorite status:', error);
    }
  };

  const toggleFavorite = async () => {
    try {
      const method = isFavorite ? 'DELETE' : 'POST';
      const response = await fetch(`${API_BASE_URL}/user/favorites/matches/${matchId}`, {
        method
      });
      
      if (response.ok) {
        setIsFavorite(!isFavorite);
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  const playNotificationSound = () => {
    if (audioRef.current) {
      audioRef.current.play().catch(console.error);
    }
  };

  const shareMatch = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `${match.home_team_name} vs ${match.away_team_name}`,
          text: `Watch live: ${match.home_score}-${match.away_score}`,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback to clipboard
      navigator.clipboard.writeText(window.location.href);
    }
  };

  // Computed values
  const matchStatusInfo = () => {
    if (!match) return {};

    const now = new Date();
    const matchDate = new Date(match.match_date);
    
    switch (match.status) {
      case 'upcoming':
        return {
          status: 'Upcoming',
          color: 'blue',
          icon: Calendar,
          time: matchDate > now ? `${Math.ceil((matchDate - now) / (1000 * 60 * 60))}h` : 'Soon'
        };
      case 'live':
        return {
          status: 'LIVE',
          color: 'red',
          icon: Activity,
          time: `${match.minute || 0}'`,
          isLive: true
        };
      case 'halftime':
        return {
          status: 'Half Time',
          color: 'yellow',
          icon: Pause,
          time: 'HT'
        };
      case 'finished':
        return {
          status: 'Full Time',
          color: 'green',
          icon: CheckCircle,
          time: 'FT'
        };
      default:
        return {
          status: 'Unknown',
          color: 'gray',
          icon: Clock,
          time: '--'
        };
    }
  };

  const statsCardsData = () => {
    if (!matchStats) return [];

    return [
      {
        id: 'home_possession',
        icon: BarChart3,
        label: 'Possession',
        value: `${matchStats.home_possession || 0}%`,
        subMetrics: [
          { label: match?.home_team_name?.substring(0, 8), value: `${matchStats.home_possession || 0}%` },
          { label: match?.away_team_name?.substring(0, 8), value: `${matchStats.away_possession || 0}%` }
        ]
      },
      {
        id: 'shots',
        icon: Target,
        label: 'Shots',
        value: (matchStats.home_shots || 0) + (matchStats.away_shots || 0),
        subMetrics: [
          { label: 'Home', value: matchStats.home_shots || 0 },
          { label: 'Away', value: matchStats.away_shots || 0 }
        ]
      },
      {
        id: 'shots_on_target',
        icon: Crosshair,
        label: 'On Target',
        value: (matchStats.home_shots_on_target || 0) + (matchStats.away_shots_on_target || 0),
        subMetrics: [
          { label: 'Home', value: matchStats.home_shots_on_target || 0 },
          { label: 'Away', value: matchStats.away_shots_on_target || 0 }
        ]
      },
      {
        id: 'cards',
        icon: Hash,
        label: 'Cards',
        value: (matchStats.home_yellow_cards || 0) + (matchStats.away_yellow_cards || 0) + (matchStats.home_red_cards || 0) + (matchStats.away_red_cards || 0),
        subMetrics: [
          { label: 'Yellow', value: (matchStats.home_yellow_cards || 0) + (matchStats.away_yellow_cards || 0) },
          { label: 'Red', value: (matchStats.home_red_cards || 0) + (matchStats.away_red_cards || 0) }
        ]
      }
    ];
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'events', label: 'Events', icon: Flag },
    { id: 'lineups', label: 'Lineups', icon: Users },
    { id: 'stats', label: 'Statistics', icon: BarChart3 },
    { id: 'predictions', label: 'AI Predictions', icon: Brain }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <Loading type="card" context="live" size="large" message="Loading live match data..." />
      </div>
    );
  }

  if (error || !match) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-8 rounded-xl text-center">
            <Activity className="mx-auto mb-4 text-red-500" size={48} />
            <h2 className="text-xl font-semibold mb-2">Match Not Found</h2>
            <p className="mb-4">{error || 'The requested match could not be found.'}</p>
            <button
              onClick={() => navigate('/live')}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Back to Live Matches
            </button>
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = matchStatusInfo();

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-4 mb-6">
              <button
                onClick={() => navigate('/live')}
                className="p-2 hover:bg-white hover:shadow-md rounded-lg transition-all"
              >
                <ArrowLeft size={20} />
              </button>
              
              <div className="flex-1">
                <div className="flex items-center gap-4 mb-2">
                  <h1 className="text-3xl font-bold text-gray-900">
                    {match.home_team_name} vs {match.away_team_name}
                  </h1>
                  
                  {statusInfo.isLive && (
                    <div className="live-indicator">
                      LIVE
                    </div>
                  )}
                </div>
                
                <div className="flex items-center gap-4 text-gray-600">
                  <div className="flex items-center gap-1">
                    <Trophy size={16} />
                    {match.league_name}
                  </div>
                  <div className="flex items-center gap-1">
                    <Calendar size={16} />
                    {new Date(match.match_date).toLocaleDateString()}
                  </div>
                  {match.stadium_name && (
                    <div className="flex items-center gap-1">
                      <MapPin size={16} />
                      {match.stadium_name}
                    </div>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-3">
                {match.status === 'live' && (
                  <button
                    onClick={() => setAutoRefresh(!autoRefresh)}
                    className={`p-2 rounded-lg transition-colors ${
                      autoRefresh
                        ? 'bg-green-100 text-green-600 hover:bg-green-200'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {autoRefresh ? <Play size={20} /> : <Pause size={20} />}
                  </button>
                )}
                
                <button
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  className={`p-2 rounded-lg transition-colors ${
                    soundEnabled
                      ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {soundEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
                </button>
                
                <button
                  onClick={toggleFavorite}
                  className={`p-2 rounded-lg transition-colors ${
                    isFavorite
                      ? 'bg-yellow-100 text-yellow-600 hover:bg-yellow-200'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {isFavorite ? <BookmarkCheck size={20} /> : <Bookmark size={20} />}
                </button>
                
                <button
                  onClick={shareMatch}
                  className="p-2 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-colors"
                >
                  <Share2 size={20} />
                </button>
              </div>
            </div>
          </div>

          {/* Score Display */}
          <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
            <div className="flex items-center justify-between">
              {/* Home Team */}
              <div className="text-center flex-1">
                <button
                  onClick={() => navigate(`/teams/${match.home_team_id}`)}
                  className="group"
                >
                  <div className="w-16 h-16 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-3 group-hover:bg-blue-200 transition-colors">
                    <Users size={32} className="text-blue-600" />
                  </div>
                  <h2 className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                    {match.home_team_name}
                  </h2>
                </button>
              </div>

              {/* Score */}
              <div className="text-center px-8">
                <div className="text-6xl font-bold text-gray-900 mb-2">
                  {match.home_score !== null ? match.home_score : '-'} - {match.away_score !== null ? match.away_score : '-'}
                </div>
                
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-${statusInfo.color}-100 text-${statusInfo.color}-700`}>
                  <statusInfo.icon size={16} />
                  {statusInfo.status}
                  {statusInfo.time && (
                    <span className="ml-1">{statusInfo.time}</span>
                  )}
                </div>

                {predictions && (
                  <div className="mt-4">
                    <div className="text-xs text-gray-500 mb-1">AI Prediction</div>
                    <div className="text-sm font-medium text-purple-600">
                      {predictions.predicted_winner || 'Draw'} 
                      <span className="text-gray-500 ml-1">
                        ({(predictions.confidence * 100).toFixed(0)}%)
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Away Team */}
              <div className="text-center flex-1">
                <button
                  onClick={() => navigate(`/teams/${match.away_team_id}`)}
                  className="group"
                >
                  <div className="w-16 h-16 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-3 group-hover:bg-red-200 transition-colors">
                    <Users size={32} className="text-red-600" />
                  </div>
                  <h2 className="text-xl font-bold text-gray-900 group-hover:text-red-600 transition-colors">
                    {match.away_team_name}
                  </h2>
                </button>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          {matchStats && (
            <div className="mb-8">
              <StatsCards 
                stats={statsCardsData()}
                loading={false}
                cardSize="medium"
                columns={4}
                showTrends={false}
              />
            </div>
          )}

          {/* Tabs */}
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="flex overflow-x-auto">
              {tabs.map(tab => {
                const IconComponent = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setSelectedTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                      selectedTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    <IconComponent size={16} />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tab Content */}
          <div className="space-y-8">
            {selectedTab === 'overview' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Recent Events */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Flag size={20} />
                    Match Events
                  </h3>
                  
                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {liveEvents.slice(0, 10).map((event, index) => (
                      <div key={index} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                        <div className="w-12 text-center">
                          <span className="text-sm font-medium text-gray-600">{event.minute}'</span>
                        </div>
                        
                        <div className="w-8 flex justify-center">
                          {event.type === 'goal' && <Target size={16} className="text-green-600" />}
                          {event.type === 'yellow_card' && <Hash size={16} className="text-yellow-500" />}
                          {event.type === 'red_card' && <Hash size={16} className="text-red-500" />}
                          {event.type === 'substitution' && <Users size={16} className="text-blue-600" />}
                        </div>
                        
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{event.player_name}</div>
                          <div className="text-sm text-gray-500">{event.team_name}</div>
                        </div>
                        
                        <div className="text-sm text-gray-500 capitalize">
                          {event.type.replace('_', ' ')}
                        </div>
                      </div>
                    ))}
                    
                    {liveEvents.length === 0 && (
                      <div className="text-center py-8 text-gray-500">
                        <Flag size={24} className="mx-auto mb-2 opacity-50" />
                        <p>No events yet</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Live Commentary */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <MessageSquare size={20} />
                    Live Commentary
                  </h3>
                  
                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {commentary.map((comment, index) => (
                      <div key={index} className="p-3 border-l-4 border-blue-500 bg-blue-50">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-blue-600">{comment.minute}'</span>
                          {comment.is_important && <Star size={12} className="text-yellow-500" />}
                        </div>
                        <p className="text-gray-800">{comment.text}</p>
                      </div>
                    ))}
                    
                    {commentary.length === 0 && (
                      <div className="text-center py-8 text-gray-500">
                        <MessageSquare size={24} className="mx-auto mb-2 opacity-50" />
                        <p>No commentary available</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'events' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Match Timeline</h3>
                
                <div className="space-y-4">
                  {liveEvents.map((event, index) => (
                    <div key={index} className="flex items-center gap-6 p-4 border border-gray-200 rounded-lg">
                      <div className="w-16 text-center">
                        <div className="text-lg font-bold text-gray-900">{event.minute}'</div>
                        {event.added_time && (
                          <div className="text-xs text-gray-500">+{event.added_time}</div>
                        )}
                      </div>
                      
                      <div className="w-8 flex justify-center">
                        {event.type === 'goal' && <Target size={20} className="text-green-600" />}
                        {event.type === 'yellow_card' && <Hash size={20} className="text-yellow-500" />}
                        {event.type === 'red_card' && <Hash size={20} className="text-red-500" />}
                        {event.type === 'substitution' && <Users size={20} className="text-blue-600" />}
                      </div>
                      
                      <div className="flex-1">
                        <div className="font-semibold text-gray-900 text-lg">{event.player_name}</div>
                        <div className="text-gray-600">{event.team_name}</div>
                        {event.description && (
                          <div className="text-sm text-gray-500 mt-1">{event.description}</div>
                        )}
                      </div>
                      
                      <div className="text-right">
                        <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                          event.type === 'goal' ? 'bg-green-100 text-green-700' :
                          event.type === 'yellow_card' ? 'bg-yellow-100 text-yellow-700' :
                          event.type === 'red_card' ? 'bg-red-100 text-red-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {event.type.replace('_', ' ').toUpperCase()}
                        </div>
                        {event.score_after && (
                          <div className="text-sm text-gray-500 mt-1">{event.score_after}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'lineups' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Home Team Lineup */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Users size={20} />
                    {match.home_team_name}
                  </h3>
                  
                  <div className="space-y-3">
                    {lineups.home.map((player, index) => (
                      <div key={player.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-sm font-bold text-blue-600">{player.jersey_number}</span>
                        </div>
                        
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{player.name}</div>
                          <div className="text-sm text-gray-500">{player.position}</div>
                        </div>
                        
                        <div className="flex gap-1">
                          {player.yellow_cards > 0 && (
                            <div className="w-4 h-4 bg-yellow-400 rounded-sm"></div>
                          )}
                          {player.red_cards > 0 && (
                            <div className="w-4 h-4 bg-red-500 rounded-sm"></div>
                          )}
                          {player.is_substituted && (
                            <div className="text-xs text-gray-500">SUB</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Away Team Lineup */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Users size={20} />
                    {match.away_team_name}
                  </h3>
                  
                  <div className="space-y-3">
                    {lineups.away.map((player, index) => (
                      <div key={player.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                        <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                          <span className="text-sm font-bold text-red-600">{player.jersey_number}</span>
                        </div>
                        
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{player.name}</div>
                          <div className="text-sm text-gray-500">{player.position}</div>
                        </div>
                        
                        <div className="flex gap-1">
                          {player.yellow_cards > 0 && (
                            <div className="w-4 h-4 bg-yellow-400 rounded-sm"></div>
                          )}
                          {player.red_cards > 0 && (
                            <div className="w-4 h-4 bg-red-500 rounded-sm"></div>
                          )}
                          {player.is_substituted && (
                            <div className="text-xs text-gray-500">SUB</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'stats' && matchStats && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Match Statistics</h3>
                
                <div className="space-y-6">
                  {/* Possession Chart */}
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Possession</h4>
                    <div className="flex items-center gap-4">
                      <span className="text-sm font-medium text-blue-600">{match.home_team_name}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-4">
                        <div 
                          className="bg-blue-600 h-4 rounded-full transition-all duration-500"
                          style={{ width: `${matchStats.home_possession || 0}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-red-600">{match.away_team_name}</span>
                    </div>
                    <div className="flex justify-between text-sm text-gray-600 mt-1">
                      <span>{matchStats.home_possession || 0}%</span>
                      <span>{matchStats.away_possession || 0}%</span>
                    </div>
                  </div>

                  {/* Stats Comparison */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Shots</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-blue-600">{matchStats.home_shots || 0}</span>
                          <span className="text-sm text-gray-600">Total Shots</span>
                          <span className="text-red-600">{matchStats.away_shots || 0}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-blue-600">{matchStats.home_shots_on_target || 0}</span>
                          <span className="text-sm text-gray-600">On Target</span>
                          <span className="text-red-600">{matchStats.away_shots_on_target || 0}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Cards</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-blue-600">{matchStats.home_yellow_cards || 0}</span>
                          <span className="text-sm text-gray-600">Yellow Cards</span>
                          <span className="text-red-600">{matchStats.away_yellow_cards || 0}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-blue-600">{matchStats.home_red_cards || 0}</span>
                          <span className="text-sm text-gray-600">Red Cards</span>
                          <span className="text-red-600">{matchStats.away_red_cards || 0}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'predictions' && predictions && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                  <Brain size={20} />
                  AI Match Predictions
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-6 border border-gray-200 rounded-lg">
                    <div className="text-3xl font-bold text-green-600 mb-2">
                      {(predictions.home_win_prob * 100).toFixed(0)}%
                    </div>
                    <div className="text-gray-600">Home Win</div>
                    <div className="text-sm text-gray-500 mt-1">{match.home_team_name}</div>
                  </div>
                  
                  <div className="text-center p-6 border border-gray-200 rounded-lg">
                    <div className="text-3xl font-bold text-yellow-600 mb-2">
                      {(predictions.draw_prob * 100).toFixed(0)}%
                    </div>
                    <div className="text-gray-600">Draw</div>
                    <div className="text-sm text-gray-500 mt-1">Equal Result</div>
                  </div>
                  
                  <div className="text-center p-6 border border-gray-200 rounded-lg">
                    <div className="text-3xl font-bold text-red-600 mb-2">
                      {(predictions.away_win_prob * 100).toFixed(0)}%
                    </div>
                    <div className="text-gray-600">Away Win</div>
                    <div className="text-sm text-gray-500 mt-1">{match.away_team_name}</div>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-purple-50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap size={16} className="text-purple-600" />
                    <span className="font-semibold text-purple-900">AI Confidence</span>
                  </div>
                  <div className="text-lg font-bold text-purple-600">
                    {(predictions.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-purple-700 mt-1">
                    Model prediction confidence level
                  </div>
                </div>

                {predictions.predicted_score && (
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Target size={16} className="text-blue-600" />
                      <span className="font-semibold text-blue-900">Predicted Score</span>
                    </div>
                    <div className="text-lg font-bold text-blue-600">
                      {predictions.predicted_score}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Audio element for notifications */}
          <audio
            ref={audioRef}
            preload="auto"
            style={{ display: 'none' }}
          >
            <source src="/notification.mp3" type="audio/mpeg" />
            <source src="/notification.wav" type="audio/wav" />
          </audio>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default LiveMatch;