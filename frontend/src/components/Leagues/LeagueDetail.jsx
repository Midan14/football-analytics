import {
    Activity,
    ArrowLeft,
    Award,
    BarChart3,
    Bookmark,
    BookmarkCheck,
    Calendar,
    Clock,
    Download,
    ExternalLink,
    Filter,
    Flag,
    Globe,
    Info,
    Search,
    Share2,
    Star,
    Target,
    Trophy,
    Users
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

// Import components
import ErrorBoundary from '../Common/ErrorBoundary';
import Loading from '../Common/Loading';
import StatsCards from '../Dashboard/StatsCards';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const LeagueDetail = () => {
  // URL Parameters
  const { leagueId } = useParams();
  const navigate = useNavigate();

  // Estados principales
  const [league, setLeague] = useState(null);
  const [leagueStats, setLeagueStats] = useState(null);
  const [teams, setTeams] = useState([]);
  const [recentMatches, setRecentMatches] = useState([]);
  const [upcomingMatches, setUpcomingMatches] = useState([]);
  const [standings, setStandings] = useState([]);
  const [topScorers, setTopScorers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isFavorite, setIsFavorite] = useState(false);

  // Estados de configuración
  const [activeTab, setActiveTab] = useState('overview'); // overview, teams, matches, standings, statistics
  const [selectedSeason, setSelectedSeason] = useState('2024-25');
  const [matchFilter, setMatchFilter] = useState('all'); // all, home, away, recent, upcoming

  // Efectos
  useEffect(() => {
    if (leagueId) {
      fetchLeagueData();
      checkIfFavorite();
    }
  }, [leagueId, selectedSeason]);

  // API Functions
  const fetchLeagueData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [
        leagueResponse,
        statsResponse,
        teamsResponse,
        matchesResponse,
        standingsResponse,
        scorersResponse
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/leagues/${leagueId}`),
        fetch(`${API_BASE_URL}/leagues/${leagueId}/statistics?season=${selectedSeason}`),
        fetch(`${API_BASE_URL}/leagues/${leagueId}/teams?season=${selectedSeason}`),
        fetch(`${API_BASE_URL}/leagues/${leagueId}/matches?season=${selectedSeason}&limit=10`),
        fetch(`${API_BASE_URL}/leagues/${leagueId}/standings?season=${selectedSeason}`),
        fetch(`${API_BASE_URL}/leagues/${leagueId}/top-scorers?season=${selectedSeason}&limit=10`)
      ]);

      if (!leagueResponse.ok) {
        throw new Error('League not found');
      }

      const leagueData = await leagueResponse.json();
      const statsData = statsResponse.ok ? await statsResponse.json() : null;
      const teamsData = teamsResponse.ok ? await teamsResponse.json() : null;
      const matchesData = matchesResponse.ok ? await matchesResponse.json() : null;
      const standingsData = standingsResponse.ok ? await standingsResponse.json() : null;
      const scorersData = scorersResponse.ok ? await scorersResponse.json() : null;

      setLeague(leagueData.league);
      setLeagueStats(statsData?.statistics);
      setTeams(teamsData?.teams || []);
      
      if (matchesData?.matches) {
        const now = new Date();
        setRecentMatches(matchesData.matches.filter(m => new Date(m.match_date) < now));
        setUpcomingMatches(matchesData.matches.filter(m => new Date(m.match_date) >= now));
      }
      
      setStandings(standingsData?.standings || []);
      setTopScorers(scorersData?.players || []);

    } catch (error) {
      console.error('Error fetching league data:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const checkIfFavorite = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/user/favorites/leagues/${leagueId}`);
      setIsFavorite(response.ok);
    } catch (error) {
      console.error('Error checking favorite status:', error);
    }
  };

  const toggleFavorite = async () => {
    try {
      const method = isFavorite ? 'DELETE' : 'POST';
      const response = await fetch(`${API_BASE_URL}/user/favorites/leagues/${leagueId}`, {
        method
      });
      
      if (response.ok) {
        setIsFavorite(!isFavorite);
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  // Computed values
  const leagueStatCards = useMemo(() => {
    if (!leagueStats) return [];

    return [
      {
        id: 'total_teams',
        icon: Users,
        label: 'Teams',
        value: teams.length,
        description: `${selectedSeason} season`,
        type: 'teams'
      },
      {
        id: 'total_matches',
        icon: Activity,
        label: 'Matches Played',
        value: leagueStats.total_matches || 0,
        description: 'This season',
        type: 'matches'
      },
      {
        id: 'avg_goals',
        icon: Target,
        label: 'Avg Goals/Match',
        value: leagueStats.avg_goals_per_match?.toFixed(2) || 'N/A',
        description: 'Season average',
        type: 'analytics'
      },
      {
        id: 'competitive_balance',
        icon: BarChart3,
        label: 'Competitive Balance',
        value: leagueStats.competitive_balance ? `${(leagueStats.competitive_balance * 100).toFixed(1)}%` : 'N/A',
        description: 'League parity index',
        type: 'analytics'
      }
    ];
  }, [leagueStats, teams, selectedSeason]);

  const goalsData = useMemo(() => {
    if (!leagueStats?.goals_by_round) return [];
    
    return leagueStats.goals_by_round.map(round => ({
      round: `Round ${round.round}`,
      goals: round.total_goals,
      avg_goals: (round.total_goals / round.matches).toFixed(2)
    }));
  }, [leagueStats]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
  };

  const handleMatchClick = (matchId) => {
    navigate(`/matches/${matchId}`);
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'teams', label: 'Teams', icon: Users },
    { id: 'matches', label: 'Matches', icon: Activity },
    { id: 'standings', label: 'Standings', icon: Trophy },
    { id: 'statistics', label: 'Statistics', icon: BarChart3 }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <Loading type="card" context="leagues" size="large" message="Loading league details..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-8 rounded-xl text-center">
            <Trophy className="mx-auto mb-4 text-red-500" size={48} />
            <h2 className="text-xl font-semibold mb-2">League Not Found</h2>
            <p className="mb-4">{error}</p>
            <button
              onClick={() => navigate('/leagues')}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Back to Leagues
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!league) return null;

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-4 mb-6">
              <button
                onClick={() => navigate('/leagues')}
                className="p-2 hover:bg-white hover:shadow-md rounded-lg transition-all"
              >
                <ArrowLeft size={20} />
              </button>
              
              <div className="flex-1">
                <div className="flex items-center gap-4 mb-2">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-green-600 rounded-xl flex items-center justify-center">
                    <Trophy className="text-white" size={32} />
                  </div>
                  
                  <div>
                    <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                      {league.name}
                      {league.level === 1 && <Star className="text-yellow-500" size={24} />}
                    </h1>
                    <div className="flex items-center gap-4 text-gray-600">
                      <div className="flex items-center gap-1">
                        <Flag size={16} />
                        {league.country}
                      </div>
                      <div className="flex items-center gap-1">
                        <Globe size={16} />
                        {league.confederation}
                      </div>
                      <div className="flex items-center gap-1">
                        <Award size={16} />
                        Level {league.level}
                      </div>
                      {league.gender === 'F' && (
                        <div className="px-2 py-1 bg-pink-100 text-pink-700 rounded-full text-xs font-medium">
                          Women's
                        </div>
                      )}
                    </div>
                  </div>
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
                
                <button className="p-2 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-colors">
                  <Share2 size={20} />
                </button>
                
                <button className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                  <Download size={20} />
                </button>
              </div>
            </div>

            {/* Season Selector */}
            <div className="flex items-center gap-4">
              <select
                value={selectedSeason}
                onChange={(e) => setSelectedSeason(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
              >
                <option value="2024-25">2024-25 Season</option>
                <option value="2023-24">2023-24 Season</option>
                <option value="2022-23">2022-23 Season</option>
              </select>
              
              <div className="text-sm text-gray-500">
                League Code: <span className="font-mono font-medium">{league.code}</span>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="mb-8">
            <StatsCards 
              stats={leagueStatCards}
              loading={false}
              cardSize="medium"
              columns={4}
            />
          </div>

          {/* Navigation Tabs */}
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="flex overflow-x-auto">
              {tabs.map(tab => {
                const IconComponent = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => handleTabChange(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                      activeTab === tab.id
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
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Goals Trend Chart */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                  <Target size={20} />
                  Goals Trend by Round
                </h3>
                
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={goalsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="goals" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        name="Total Goals"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="avg_goals" 
                        stroke="#10B981" 
                        strokeWidth={2}
                        name="Avg Goals/Match"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* League Information */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                  <Info size={20} />
                  League Information
                </h3>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-500">Country</div>
                      <div className="font-semibold">{league.country}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Confederation</div>
                      <div className="font-semibold">{league.confederation}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Division Level</div>
                      <div className="font-semibold">Level {league.level}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Gender</div>
                      <div className="font-semibold">{league.gender === 'M' ? 'Men\'s' : 'Women\'s'}</div>
                    </div>
                  </div>

                  {leagueStats && (
                    <div className="pt-4 border-t border-gray-200">
                      <h4 className="font-semibold text-gray-900 mb-3">Season Statistics</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Total Goals</span>
                          <span className="font-medium">{leagueStats.total_goals || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Avg Attendance</span>
                          <span className="font-medium">{leagueStats.avg_attendance?.toLocaleString() || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Yellow Cards</span>
                          <span className="font-medium">{leagueStats.total_yellow_cards || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Red Cards</span>
                          <span className="font-medium">{leagueStats.total_red_cards || 'N/A'}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'teams' && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <Users size={20} />
                  Teams ({teams.length})
                </h3>
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                    <input
                      type="text"
                      placeholder="Search teams..."
                      className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {teams.map(team => (
                  <div
                    key={team.id}
                    onClick={() => handleTeamClick(team.id)}
                    className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 cursor-pointer transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                        <Users size={20} className="text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-gray-900">{team.name}</div>
                        <div className="text-sm text-gray-500">
                          {team.stadium_name && `${team.stadium_name} • `}
                          Founded {team.founded_year}
                        </div>
                      </div>
                      <ExternalLink size={16} className="text-gray-400" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'matches' && (
            <div className="space-y-6">
              {/* Match Filter */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center gap-4">
                  <Filter size={20} className="text-gray-500" />
                  <select
                    value={matchFilter}
                    onChange={(e) => setMatchFilter(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Matches</option>
                    <option value="recent">Recent Matches</option>
                    <option value="upcoming">Upcoming Matches</option>
                  </select>
                </div>
              </div>

              {/* Recent Matches */}
              {(matchFilter === 'all' || matchFilter === 'recent') && recentMatches.length > 0 && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Clock size={20} />
                    Recent Matches
                  </h3>
                  
                  <div className="space-y-3">
                    {recentMatches.slice(0, 10).map(match => (
                      <div
                        key={match.id}
                        onClick={() => handleMatchClick(match.id)}
                        className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 flex-1">
                            <div className="text-center min-w-0 flex-1">
                              <div className="font-medium text-gray-900 truncate">{match.home_team_name}</div>
                            </div>
                            <div className="text-center px-4">
                              <div className="text-xl font-bold text-gray-900">
                                {match.home_score} - {match.away_score}
                              </div>
                              <div className="text-xs text-gray-500">
                                {match.status === 'finished' ? 'FT' : match.status}
                              </div>
                            </div>
                            <div className="text-center min-w-0 flex-1">
                              <div className="font-medium text-gray-900 truncate">{match.away_team_name}</div>
                            </div>
                          </div>
                          <div className="text-sm text-gray-500 ml-4">
                            {new Date(match.match_date).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Upcoming Matches */}
              {(matchFilter === 'all' || matchFilter === 'upcoming') && upcomingMatches.length > 0 && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <Calendar size={20} />
                    Upcoming Matches
                  </h3>
                  
                  <div className="space-y-3">
                    {upcomingMatches.slice(0, 10).map(match => (
                      <div
                        key={match.id}
                        onClick={() => handleMatchClick(match.id)}
                        className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 flex-1">
                            <div className="text-center min-w-0 flex-1">
                              <div className="font-medium text-gray-900 truncate">{match.home_team_name}</div>
                            </div>
                            <div className="text-center px-4">
                              <div className="text-sm text-gray-500">
                                vs
                              </div>
                              <div className="text-xs text-gray-400">
                                {new Date(match.match_date).toLocaleDateString()}
                              </div>
                            </div>
                            <div className="text-center min-w-0 flex-1">
                              <div className="font-medium text-gray-900 truncate">{match.away_team_name}</div>
                            </div>
                          </div>
                          <div className="text-sm text-gray-500 ml-4">
                            {new Date(match.match_date).toLocaleTimeString([], { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'standings' && standings.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                <Trophy size={20} />
                League Table
              </h3>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-semibold text-gray-700">Pos</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-700">Team</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">MP</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">W</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">D</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">L</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">GF</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">GA</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">GD</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-700">Pts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {standings.map((team, index) => (
                      <tr key={team.team_id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{index + 1}</span>
                            {index < 4 && <div className="w-2 h-2 bg-green-500 rounded-full"></div>}
                            {index >= standings.length - 3 && <div className="w-2 h-2 bg-red-500 rounded-full"></div>}
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => handleTeamClick(team.team_id)}
                            className="font-medium text-gray-900 hover:text-blue-600 transition-colors"
                          >
                            {team.team_name}
                          </button>
                        </td>
                        <td className="py-3 px-4 text-center">{team.matches_played}</td>
                        <td className="py-3 px-4 text-center">{team.wins}</td>
                        <td className="py-3 px-4 text-center">{team.draws}</td>
                        <td className="py-3 px-4 text-center">{team.losses}</td>
                        <td className="py-3 px-4 text-center">{team.goals_for}</td>
                        <td className="py-3 px-4 text-center">{team.goals_against}</td>
                        <td className="py-3 px-4 text-center">
                          <span className={team.goal_difference >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center font-bold">{team.points}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'statistics' && (
            <div className="space-y-6">
              {/* Top Scorers */}
              {topScorers.length > 0 && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <Award size={20} />
                    Top Scorers
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {topScorers.slice(0, 10).map((player, index) => (
                      <div key={player.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                        <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                          <span className="font-bold text-yellow-700">{index + 1}</span>
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">{player.name}</div>
                          <div className="text-sm text-gray-500">{player.team_name}</div>
                        </div>
                        <div className="text-xl font-bold text-blue-600">
                          {player.goals}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* League Statistics Chart */}
              {leagueStats && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <BarChart3 size={20} />
                    Season Overview
                  </h3>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600 mb-2">
                        {leagueStats.total_goals || 0}
                      </div>
                      <div className="text-sm text-gray-600">Total Goals</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-600 mb-2">
                        {leagueStats.avg_goals_per_match?.toFixed(2) || 'N/A'}
                      </div>
                      <div className="text-sm text-gray-600">Goals/Match</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-yellow-600 mb-2">
                        {leagueStats.total_yellow_cards || 0}
                      </div>
                      <div className="text-sm text-gray-600">Yellow Cards</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-red-600 mb-2">
                        {leagueStats.total_red_cards || 0}
                      </div>
                      <div className="text-sm text-gray-600">Red Cards</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default LeagueDetail;