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
    Download,
    ExternalLink,
    Eye,
    Flag,
    Hash,
    Info,
    MapPin,
    Percent,
    Play,
    Share2,
    Target,
    TrendingUp,
    Trophy,
    Users,
    XCircle,
    Zap
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    Legend,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis,
    Radar,
    RadarChart,
    ResponsiveContainer
} from 'recharts';

// Import components
import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const MatchDetail = () => {
  // URL Parameters
  const { matchId } = useParams();
  const navigate = useNavigate();

  // Estados principales
  const [match, setMatch] = useState(null);
  const [matchStats, setMatchStats] = useState(null);
  const [matchEvents, setMatchEvents] = useState([]);
  const [lineups, setLineups] = useState({ home: [], away: [] });
  const [predictions, setPredictions] = useState(null);
  const [headToHead, setHeadToHead] = useState([]);
  const [similarMatches, setSimilarMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isFavorite, setIsFavorite] = useState(false);

  // Estados de configuración
  const [selectedTab, setSelectedTab] = useState('overview'); // overview, events, lineups, stats, predictions, analysis
  const [comparisonMode, setComparisonMode] = useState('teams'); // teams, seasons, league_avg

  // Efectos
  useEffect(() => {
    if (matchId) {
      fetchMatchData();
      checkIfFavorite();
    }
  }, [matchId]);

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
        h2hResponse,
        similarResponse
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/matches/${matchId}`),
        fetch(`${API_BASE_URL}/matches/${matchId}/statistics`),
        fetch(`${API_BASE_URL}/matches/${matchId}/events`),
        fetch(`${API_BASE_URL}/matches/${matchId}/lineups`),
        fetch(`${API_BASE_URL}/matches/${matchId}/predictions`),
        fetch(`${API_BASE_URL}/matches/${matchId}/head-to-head?limit=5`),
        fetch(`${API_BASE_URL}/matches/${matchId}/similar?limit=5`)
      ]);

      if (!matchResponse.ok) {
        throw new Error('Match not found');
      }

      const matchData = await matchResponse.json();
      const statsData = statsResponse.ok ? await statsResponse.json() : null;
      const eventsData = eventsResponse.ok ? await eventsResponse.json() : null;
      const lineupsData = lineupsResponse.ok ? await lineupsResponse.json() : null;
      const predictionsData = predictionsResponse.ok ? await predictionsResponse.json() : null;
      const h2hData = h2hResponse.ok ? await h2hResponse.json() : null;
      const similarData = similarResponse.ok ? await similarResponse.json() : null;

      setMatch(matchData.match);
      setMatchStats(statsData?.statistics);
      setMatchEvents(eventsData?.events || []);
      setLineups(lineupsData?.lineups || { home: [], away: [] });
      setPredictions(predictionsData?.predictions);
      setHeadToHead(h2hData?.matches || []);
      setSimilarMatches(similarData?.matches || []);

    } catch (error) {
      console.error('Error fetching match data:', error);
      setError(error.message);
    } finally {
      setLoading(false);
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

  const shareMatch = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `${match.home_team_name} vs ${match.away_team_name}`,
          text: `Match analysis: ${match.home_score !== null ? `${match.home_score}-${match.away_score}` : 'Upcoming match'}`,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  };

  const exportMatchData = () => {
    const data = {
      match,
      statistics: matchStats,
      events: matchEvents,
      lineups,
      predictions,
      head_to_head: headToHead,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `match-${match.home_team_name}-vs-${match.away_team_name}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Computed values
  const matchStatusInfo = useMemo(() => {
    if (!match) return {};

    const now = new Date();
    const matchDate = new Date(match.match_date);
    
    switch (match.status) {
      case 'upcoming':
        return {
          status: 'Upcoming',
          color: 'blue',
          icon: Calendar,
          time: matchDate > now ? matchDate.toLocaleDateString() : 'Today',
          description: 'Match not started yet'
        };
      case 'live':
        return {
          status: 'LIVE',
          color: 'red',
          icon: Activity,
          time: `${match.minute || 0}'`,
          description: 'Match in progress',
          isLive: true
        };
      case 'halftime':
        return {
          status: 'Half Time',
          color: 'yellow',
          icon: Clock,
          time: 'HT',
          description: 'Half time break'
        };
      case 'finished':
        return {
          status: 'Full Time',
          color: 'green',
          icon: CheckCircle,
          time: 'FT',
          description: 'Match completed'
        };
      case 'postponed':
        return {
          status: 'Postponed',
          color: 'gray',
          icon: XCircle,
          time: 'PP',
          description: 'Match postponed'
        };
      case 'cancelled':
        return {
          status: 'Cancelled',
          color: 'red',
          icon: XCircle,
          time: 'CAN',
          description: 'Match cancelled'
        };
      default:
        return {
          status: 'Unknown',
          color: 'gray',
          icon: Clock,
          time: '--',
          description: 'Status unknown'
        };
    }
  }, [match]);

  const statsCardsData = useMemo(() => {
    if (!matchStats) return [];

    return [
      {
        id: 'possession',
        icon: BarChart3,
        label: 'Possession',
        value: `${matchStats.home_possession || 0}%`,
        progress: matchStats.home_possession || 0,
        progressColor: 'blue',
        subMetrics: [
          { label: match?.home_team_name?.substring(0, 8), value: `${matchStats.home_possession || 0}%` },
          { label: match?.away_team_name?.substring(0, 8), value: `${matchStats.away_possession || 0}%` }
        ]
      },
      {
        id: 'shots',
        icon: Target,
        label: 'Total Shots',
        value: (matchStats.home_shots || 0) + (matchStats.away_shots || 0),
        subMetrics: [
          { label: 'Home', value: matchStats.home_shots || 0 },
          { label: 'Away', value: matchStats.away_shots || 0 }
        ]
      },
      {
        id: 'shots_on_target',
        icon: Crosshair,
        label: 'Shots on Target',
        value: (matchStats.home_shots_on_target || 0) + (matchStats.away_shots_on_target || 0),
        subMetrics: [
          { label: 'Home', value: matchStats.home_shots_on_target || 0 },
          { label: 'Away', value: matchStats.away_shots_on_target || 0 }
        ]
      },
      {
        id: 'accuracy',
        icon: Percent,
        label: 'Shot Accuracy',
        value: matchStats.home_shots > 0 && matchStats.away_shots > 0 
          ? `${(((matchStats.home_shots_on_target + matchStats.away_shots_on_target) / (matchStats.home_shots + matchStats.away_shots)) * 100).toFixed(1)}%`
          : 'N/A',
        subMetrics: [
          { 
            label: 'Home', 
            value: matchStats.home_shots > 0 
              ? `${((matchStats.home_shots_on_target / matchStats.home_shots) * 100).toFixed(1)}%`
              : 'N/A'
          },
          { 
            label: 'Away', 
            value: matchStats.away_shots > 0 
              ? `${((matchStats.away_shots_on_target / matchStats.away_shots) * 100).toFixed(1)}%`
              : 'N/A'
          }
        ]
      }
    ];
  }, [matchStats, match]);

  const radarChartData = useMemo(() => {
    if (!matchStats) return [];

    return [
      {
        metric: 'Shots',
        Home: matchStats.home_shots || 0,
        Away: matchStats.away_shots || 0
      },
      {
        metric: 'On Target',
        Home: matchStats.home_shots_on_target || 0,
        Away: matchStats.away_shots_on_target || 0
      },
      {
        metric: 'Possession',
        Home: matchStats.home_possession || 0,
        Away: matchStats.away_possession || 0
      },
      {
        metric: 'Corners',
        Home: matchStats.home_corners || 0,
        Away: matchStats.away_corners || 0
      },
      {
        metric: 'Fouls',
        Home: matchStats.home_fouls || 0,
        Away: matchStats.away_fouls || 0
      },
      {
        metric: 'Cards',
        Home: (matchStats.home_yellow_cards || 0) + (matchStats.home_red_cards || 0),
        Away: (matchStats.away_yellow_cards || 0) + (matchStats.away_red_cards || 0)
      }
    ];
  }, [matchStats]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'events', label: 'Match Events', icon: Flag },
    { id: 'lineups', label: 'Lineups', icon: Users },
    { id: 'stats', label: 'Statistics', icon: BarChart3 },
    { id: 'predictions', label: 'AI Analysis', icon: Brain },
    { id: 'analysis', label: 'Deep Analysis', icon: TrendingUp }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <Loading type="card" context="matches" size="large" message="Loading match analysis..." />
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
            <div className="flex gap-3 justify-center">
              <button
                onClick={() => navigate('/matches')}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Back to Matches
              </button>
              <button
                onClick={() => navigate('/live')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Live Matches
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = matchStatusInfo;

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-4 mb-6">
              <button
                onClick={() => navigate(-1)}
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
                    <button
                      onClick={() => navigate(`/live/${matchId}`)}
                      className="live-indicator flex items-center gap-1"
                    >
                      <Play size={12} />
                      WATCH LIVE
                    </button>
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
                  <div className="flex items-center gap-1">
                    <Clock size={16} />
                    {match.kickoff_time && new Date(`2000-01-01T${match.kickoff_time}`).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
                
                <button
                  onClick={exportMatchData}
                  className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Download size={20} />
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
                  <div className="w-20 h-20 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-200 transition-colors">
                    <Users size={40} className="text-blue-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                    {match.home_team_name}
                  </h2>
                  <div className="text-sm text-gray-500 mt-1">Home</div>
                </button>
              </div>

              {/* Score & Status */}
              <div className="text-center px-8">
                {match.status === 'upcoming' ? (
                  <div className="text-center">
                    <div className="text-4xl font-bold text-gray-400 mb-3">
                      vs
                    </div>
                    <div className="text-sm text-gray-500 mb-2">
                      {new Date(match.match_date).toLocaleDateString()}
                    </div>
                    {match.kickoff_time && (
                      <div className="text-lg font-medium text-gray-700">
                        {new Date(`2000-01-01T${match.kickoff_time}`).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-6xl font-bold text-gray-900 mb-3">
                    {match.home_score !== null ? match.home_score : '-'} - {match.away_score !== null ? match.away_score : '-'}
                  </div>
                )}
                
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-${statusInfo.color}-100 text-${statusInfo.color}-700`}>
                  <statusInfo.icon size={16} />
                  {statusInfo.status}
                  {statusInfo.time && statusInfo.status !== 'Upcoming' && (
                    <span className="ml-1">{statusInfo.time}</span>
                  )}
                </div>

                <div className="text-xs text-gray-500 mt-2">
                  {statusInfo.description}
                </div>

                {/* Result Analysis */}
                {match.status === 'finished' && match.home_score !== null && match.away_score !== null && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <div className="text-sm font-medium text-gray-700">
                      {match.home_score > match.away_score ? `${match.home_team_name} Win` :
                       match.home_score < match.away_score ? `${match.away_team_name} Win` :
                       'Draw'}
                    </div>
                    {predictions && (
                      <div className="text-xs text-gray-500 mt-1">
                        AI predicted: {predictions.predicted_result} ({(predictions.confidence * 100).toFixed(0)}% confidence)
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Away Team */}
              <div className="text-center flex-1">
                <button
                  onClick={() => navigate(`/teams/${match.away_team_id}`)}
                  className="group"
                >
                  <div className="w-20 h-20 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:bg-red-200 transition-colors">
                    <Users size={40} className="text-red-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 group-hover:text-red-600 transition-colors">
                    {match.away_team_name}
                  </h2>
                  <div className="text-sm text-gray-500 mt-1">Away</div>
                </button>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          {matchStats && match.status !== 'upcoming' && (
            <div className="mb-8">
              <StatsCards 
                stats={statsCardsData}
                loading={false}
                cardSize="medium"
                columns={4}
                showTrends={false}
              />
            </div>
          )}

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <button
              onClick={() => navigate(`/analytics/comparison?team1=${match.home_team_id}&team2=${match.away_team_id}`)}
              className="p-4 bg-white rounded-xl shadow-lg hover:shadow-xl transition-all text-left group"
            >
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                  <BarChart3 size={24} className="text-blue-600" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900">Compare Teams</div>
                  <div className="text-sm text-gray-500">Deep statistical analysis</div>
                </div>
                <ExternalLink size={16} className="text-gray-400 ml-auto" />
              </div>
            </button>

            <button
              onClick={() => navigate(`/leagues/${match.league_id}`)}
              className="p-4 bg-white rounded-xl shadow-lg hover:shadow-xl transition-all text-left group"
            >
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center group-hover:bg-green-200 transition-colors">
                  <Trophy size={24} className="text-green-600" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900">League Table</div>
                  <div className="text-sm text-gray-500">See standings & stats</div>
                </div>
                <ExternalLink size={16} className="text-gray-400 ml-auto" />
              </div>
            </button>

            <button
              onClick={() => navigate(`/predictions?home_team=${match.home_team_id}&away_team=${match.away_team_id}`)}
              className="p-4 bg-white rounded-xl shadow-lg hover:shadow-xl transition-all text-left group"
            >
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center group-hover:bg-purple-200 transition-colors">
                  <Brain size={24} className="text-purple-600" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900">AI Predictions</div>
                  <div className="text-sm text-gray-500">Similar matches analysis</div>
                </div>
                <ExternalLink size={16} className="text-gray-400 ml-auto" />
              </div>
            </button>
          </div>

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
                {/* Match Info */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Info size={20} />
                    Match Information
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm text-gray-500">Date & Time</div>
                        <div className="font-semibold">
                          {new Date(match.match_date).toLocaleDateString()}
                          {match.kickoff_time && ` • ${new Date(`2000-01-01T${match.kickoff_time}`).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Competition</div>
                        <div className="font-semibold">{match.league_name}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Season</div>
                        <div className="font-semibold">{match.season || '2024-25'}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Matchday</div>
                        <div className="font-semibold">Round {match.round || 'N/A'}</div>
                      </div>
                    </div>

                    {match.stadium_name && (
                      <div className="pt-4 border-t border-gray-200">
                        <div className="text-sm text-gray-500">Venue</div>
                        <div className="font-semibold">{match.stadium_name}</div>
                        {match.attendance && (
                          <div className="text-sm text-gray-600 mt-1">
                            Attendance: {match.attendance.toLocaleString()}
                          </div>
                        )}
                      </div>
                    )}

                    {match.referee && (
                      <div className="pt-4 border-t border-gray-200">
                        <div className="text-sm text-gray-500">Referee</div>
                        <div className="font-semibold">{match.referee}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Recent Form */}
                {headToHead.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <Clock size={20} />
                      Head to Head (Last 5)
                    </h3>
                    
                    <div className="space-y-3">
                      {headToHead.map((h2hMatch, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                          <div className="flex-1">
                            <div className="font-medium text-gray-900">
                              {h2hMatch.home_team_name} vs {h2hMatch.away_team_name}
                            </div>
                            <div className="text-sm text-gray-500">
                              {new Date(h2hMatch.match_date).toLocaleDateString()} • {h2hMatch.league_name}
                            </div>
                          </div>
                          
                          <div className="text-center">
                            <div className="font-bold text-lg text-gray-900">
                              {h2hMatch.home_score} - {h2hMatch.away_score}
                            </div>
                            <div className="text-xs text-gray-500">
                              {h2hMatch.home_score > h2hMatch.away_score ? 'H' :
                               h2hMatch.home_score < h2hMatch.away_score ? 'A' : 'D'}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {selectedTab === 'events' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-6">Match Events</h3>
                
                {matchEvents.length > 0 ? (
                  <div className="space-y-4">
                    {matchEvents.map((event, index) => (
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
                          {event.type === 'penalty' && <Crosshair size={20} className="text-purple-600" />}
                        </div>
                        
                        <div className="flex-1">
                          <div className="font-semibold text-gray-900 text-lg">{event.player_name}</div>
                          <div className="text-gray-600">{event.team_name}</div>
                          {event.description && (
                            <div className="text-sm text-gray-500 mt-1">{event.description}</div>
                          )}
                          {event.assist_player && (
                            <div className="text-sm text-blue-600 mt-1">Assist: {event.assist_player}</div>
                          )}
                        </div>
                        
                        <div className="text-right">
                          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                            event.type === 'goal' ? 'bg-green-100 text-green-700' :
                            event.type === 'yellow_card' ? 'bg-yellow-100 text-yellow-700' :
                            event.type === 'red_card' ? 'bg-red-100 text-red-700' :
                            event.type === 'penalty' ? 'bg-purple-100 text-purple-700' :
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
                ) : (
                  <div className="text-center py-12">
                    <Flag size={48} className="mx-auto text-gray-300 mb-4" />
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">No events recorded</h4>
                    <p className="text-gray-600">
                      {match.status === 'upcoming' 
                        ? 'Events will appear here during the match'
                        : 'No match events are available for this game'
                      }
                    </p>
                  </div>
                )}
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
                  
                  {lineups.home.length > 0 ? (
                    <div className="space-y-3">
                      <div className="text-sm font-medium text-gray-600 mb-3">Starting XI</div>
                      {lineups.home.filter(p => p.is_starter).map((player, index) => (
                        <div key={player.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                            <span className="text-sm font-bold text-blue-600">{player.jersey_number}</span>
                          </div>
                          
                          <div className="flex-1">
                            <div className="font-medium text-gray-900">{player.name}</div>
                            <div className="text-sm text-gray-500">{player.position}</div>
                          </div>
                          
                          <div className="flex gap-1">
                            {player.goals > 0 && (
                              <div className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                                {player.goals}G
                              </div>
                            )}
                            {player.yellow_cards > 0 && (
                              <div className="w-4 h-4 bg-yellow-400 rounded-sm"></div>
                            )}
                            {player.red_cards > 0 && (
                              <div className="w-4 h-4 bg-red-500 rounded-sm"></div>
                            )}
                            {player.is_substituted && (
                              <div className="text-xs text-gray-500">SUB {player.substitution_minute}'</div>
                            )}
                          </div>
                        </div>
                      ))}
                      
                      {lineups.home.filter(p => !p.is_starter).length > 0 && (
                        <>
                          <div className="text-sm font-medium text-gray-600 mt-6 mb-3">Substitutes</div>
                          {lineups.home.filter(p => !p.is_starter).map((player, index) => (
                            <div key={player.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                              <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                                <span className="text-sm font-bold text-gray-600">{player.jersey_number}</span>
                              </div>
                              
                              <div className="flex-1">
                                <div className="font-medium text-gray-900">{player.name}</div>
                                <div className="text-sm text-gray-500">{player.position}</div>
                              </div>
                              
                              {player.substitution_minute && (
                                <div className="text-xs text-blue-600">IN {player.substitution_minute}'</div>
                              )}
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Users size={24} className="mx-auto mb-2 opacity-50" />
                      <p>Lineup not available</p>
                    </div>
                  )}
                </div>

                {/* Away Team Lineup */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Users size={20} />
                    {match.away_team_name}
                  </h3>
                  
                  {lineups.away.length > 0 ? (
                    <div className="space-y-3">
                      <div className="text-sm font-medium text-gray-600 mb-3">Starting XI</div>
                      {lineups.away.filter(p => p.is_starter).map((player, index) => (
                        <div key={player.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                          <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                            <span className="text-sm font-bold text-red-600">{player.jersey_number}</span>
                          </div>
                          
                          <div className="flex-1">
                            <div className="font-medium text-gray-900">{player.name}</div>
                            <div className="text-sm text-gray-500">{player.position}</div>
                          </div>
                          
                          <div className="flex gap-1">
                            {player.goals > 0 && (
                              <div className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                                {player.goals}G
                              </div>
                            )}
                            {player.yellow_cards > 0 && (
                              <div className="w-4 h-4 bg-yellow-400 rounded-sm"></div>
                            )}
                            {player.red_cards > 0 && (
                              <div className="w-4 h-4 bg-red-500 rounded-sm"></div>
                            )}
                            {player.is_substituted && (
                              <div className="text-xs text-gray-500">SUB {player.substitution_minute}'</div>
                            )}
                          </div>
                        </div>
                      ))}
                      
                      {lineups.away.filter(p => !p.is_starter).length > 0 && (
                        <>
                          <div className="text-sm font-medium text-gray-600 mt-6 mb-3">Substitutes</div>
                          {lineups.away.filter(p => !p.is_starter).map((player, index) => (
                            <div key={player.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                              <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                                <span className="text-sm font-bold text-gray-600">{player.jersey_number}</span>
                              </div>
                              
                              <div className="flex-1">
                                <div className="font-medium text-gray-900">{player.name}</div>
                                <div className="text-sm text-gray-500">{player.position}</div>
                              </div>
                              
                              {player.substitution_minute && (
                                <div className="text-xs text-red-600">IN {player.substitution_minute}'</div>
                              )}
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Users size={24} className="mx-auto mb-2 opacity-50" />
                      <p>Lineup not available</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedTab === 'stats' && matchStats && (
              <div className="space-y-8">
                {/* Radar Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Performance Comparison</h3>
                  
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={radarChartData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis />
                        <Radar
                          name={match.home_team_name}
                          dataKey="Home"
                          stroke="#3B82F6"
                          fill="#3B82F6"
                          fillOpacity={0.1}
                          strokeWidth={2}
                        />
                        <Radar
                          name={match.away_team_name}
                          dataKey="Away"
                          stroke="#EF4444"
                          fill="#EF4444"
                          fillOpacity={0.1}
                          strokeWidth={2}
                        />
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Detailed Statistics */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6">Detailed Statistics</h3>
                  
                  <div className="space-y-6">
                    {/* Attack */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        <Target size={16} />
                        Attack
                      </h4>
                      <div className="grid grid-cols-3 gap-6 text-sm">
                        <div className="text-center">
                          <div className="font-semibold text-blue-600 text-xl">{matchStats.home_shots || 0}</div>
                          <div className="text-gray-600">Shots</div>
                        </div>
                        <div className="text-center">
                          <div className="text-gray-500">vs</div>
                        </div>
                        <div className="text-center">
                          <div className="font-semibold text-red-600 text-xl">{matchStats.away_shots || 0}</div>
                          <div className="text-gray-600">Shots</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-6 text-sm mt-3">
                        <div className="text-center">
                          <div className="font-semibold text-blue-600">{matchStats.home_shots_on_target || 0}</div>
                          <div className="text-gray-600">On Target</div>
                        </div>
                        <div className="text-center">
                          <div className="text-gray-500">vs</div>
                        </div>
                        <div className="text-center">
                          <div className="font-semibold text-red-600">{matchStats.away_shots_on_target || 0}</div>
                          <div className="text-gray-600">On Target</div>
                        </div>
                      </div>
                    </div>

                    {/* Possession */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        <BarChart3 size={16} />
                        Possession
                      </h4>
                      <div className="flex items-center gap-4">
                        <span className="text-sm font-medium text-blue-600 w-20">{matchStats.home_possession || 0}%</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-4">
                          <div 
                            className="bg-blue-600 h-4 rounded-full transition-all duration-500"
                            style={{ width: `${matchStats.home_possession || 0}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-red-600 w-20 text-right">{matchStats.away_possession || 0}%</span>
                      </div>
                    </div>

                    {/* Other Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-3">Discipline</h4>
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
                          <div className="flex justify-between items-center">
                            <span className="text-blue-600">{matchStats.home_fouls || 0}</span>
                            <span className="text-sm text-gray-600">Fouls</span>
                            <span className="text-red-600">{matchStats.away_fouls || 0}</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold text-gray-900 mb-3">Set Pieces</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-blue-600">{matchStats.home_corners || 0}</span>
                            <span className="text-sm text-gray-600">Corners</span>
                            <span className="text-red-600">{matchStats.away_corners || 0}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-blue-600">{matchStats.home_free_kicks || 0}</span>
                            <span className="text-sm text-gray-600">Free Kicks</span>
                            <span className="text-red-600">{matchStats.away_free_kicks || 0}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-blue-600">{matchStats.home_offsides || 0}</span>
                            <span className="text-sm text-gray-600">Offsides</span>
                            <span className="text-red-600">{matchStats.away_offsides || 0}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'predictions' && predictions && (
              <div className="space-y-8">
                {/* AI Predictions */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <Brain size={20} />
                    AI Match Analysis
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div className="text-center p-6 border border-gray-200 rounded-lg">
                      <div className="text-4xl font-bold text-green-600 mb-2">
                        {(predictions.home_win_prob * 100).toFixed(0)}%
                      </div>
                      <div className="text-gray-600 font-medium">{match.home_team_name} Win</div>
                      <div className="text-sm text-gray-500 mt-1">Home Victory</div>
                    </div>
                    
                    <div className="text-center p-6 border border-gray-200 rounded-lg">
                      <div className="text-4xl font-bold text-yellow-600 mb-2">
                        {(predictions.draw_prob * 100).toFixed(0)}%
                      </div>
                      <div className="text-gray-600 font-medium">Draw</div>
                      <div className="text-sm text-gray-500 mt-1">Equal Result</div>
                    </div>
                    
                    <div className="text-center p-6 border border-gray-200 rounded-lg">
                      <div className="text-4xl font-bold text-red-600 mb-2">
                        {(predictions.away_win_prob * 100).toFixed(0)}%
                      </div>
                      <div className="text-gray-600 font-medium">{match.away_team_name} Win</div>
                      <div className="text-sm text-gray-500 mt-1">Away Victory</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Zap size={16} className="text-purple-600" />
                        <span className="font-semibold text-purple-900">Model Confidence</span>
                      </div>
                      <div className="text-2xl font-bold text-purple-600">
                        {(predictions.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-purple-700 mt-1">
                        AI prediction confidence level
                      </div>
                    </div>

                    {predictions.predicted_score && (
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Target size={16} className="text-blue-600" />
                          <span className="font-semibold text-blue-900">Predicted Score</span>
                        </div>
                        <div className="text-2xl font-bold text-blue-600">
                          {predictions.predicted_score}
                        </div>
                        <div className="text-sm text-blue-700 mt-1">
                          Most likely final score
                        </div>
                      </div>
                    )}
                  </div>

                  {match.status === 'finished' && (
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-2">Prediction Accuracy</h4>
                      <div className="text-sm text-gray-600">
                        {predictions.was_correct ? (
                          <div className="flex items-center gap-2 text-green-600">
                            <CheckCircle size={16} />
                            AI prediction was correct!
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-red-600">
                            <XCircle size={16} />
                            AI prediction was incorrect
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedTab === 'analysis' && (
              <div className="space-y-8">
                {/* Similar Matches */}
                {similarMatches.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                      <TrendingUp size={20} />
                      Similar Matches Analysis
                    </h3>
                    
                    <div className="space-y-4">
                      {similarMatches.map((similar, index) => (
                        <div key={index} className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="font-medium text-gray-900">
                                {similar.home_team_name} vs {similar.away_team_name}
                              </div>
                              <div className="text-sm text-gray-500">
                                {new Date(similar.match_date).toLocaleDateString()} • {similar.league_name}
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <div className="font-bold text-lg text-gray-900">
                                {similar.home_score} - {similar.away_score}
                              </div>
                              <div className="text-xs text-gray-500">Final Score</div>
                            </div>
                            
                            <div className="text-right">
                              <div className="text-sm text-purple-600 font-medium">
                                {(similar.similarity_score * 100).toFixed(0)}% similar
                              </div>
                              <div className="text-xs text-gray-500">Match similarity</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Match Insights */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <Eye size={20} />
                    Key Insights
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 border border-gray-200 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-3">Tactical Analysis</h4>
                      <div className="space-y-2 text-sm text-gray-600">
                        {matchStats && (
                          <>
                            <div>
                              • {matchStats.home_possession > 60 ? `${match.home_team_name} dominated possession` : 
                                  matchStats.away_possession > 60 ? `${match.away_team_name} dominated possession` :
                                  'Balanced possession between both teams'}
                            </div>
                            <div>
                              • {matchStats.home_shots > matchStats.away_shots * 1.5 ? `${match.home_team_name} created more chances` :
                                  matchStats.away_shots > matchStats.home_shots * 1.5 ? `${match.away_team_name} created more chances` :
                                  'Even shot distribution'}
                            </div>
                            <div>
                              • {(matchStats.home_yellow_cards + matchStats.away_yellow_cards) > 6 ? 'Physical and intense match' :
                                  (matchStats.home_yellow_cards + matchStats.away_yellow_cards) < 3 ? 'Clean and fair match' :
                                  'Normal level of physicality'}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="p-4 border border-gray-200 rounded-lg">
                      <h4 className="font-semibold text-gray-900 mb-3">Performance Rating</h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm text-gray-600">{match.home_team_name}</span>
                            <span className="text-sm font-medium">8.2/10</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div className="bg-blue-600 h-2 rounded-full" style={{ width: '82%' }}></div>
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm text-gray-600">{match.away_team_name}</span>
                            <span className="text-sm font-medium">7.5/10</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div className="bg-red-600 h-2 rounded-full" style={{ width: '75%' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default MatchDetail;