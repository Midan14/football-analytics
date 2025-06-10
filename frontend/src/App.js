import { Suspense, lazy, useEffect, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';

// =============================================================================
// CONTEXTOS Y PROVIDERS
// =============================================================================
import { AppProvider } from './context/AppContext';
import { AuthProvider } from './context/AuthContext';

// =============================================================================
// HOOKS PERSONALIZADOS
// =============================================================================
import { useLocalStorage } from './hooks/useLocalStorage';

// =============================================================================
// SERVICIOS
// =============================================================================
import authService from './services/auth';
import webSocketService from './services/websocket';

// =============================================================================
// COMPONENTES DE LAYOUT
// =============================================================================
import ErrorFallback from './components/Common/ErrorFallback';
import LoadingScreen from './components/Common/LoadingScreen';
import NotificationContainer from './components/Common/NotificationContainer';
import Footer from './components/Layout/Footer';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';

// =============================================================================
// COMPONENTES LAZY LOADING (Carga perezosa para mejor rendimiento)
// =============================================================================
const Dashboard = lazy(() => import('./pages/Dashboard'));
const MatchList = lazy(() => import('./components/Matches/MatchList'));
const MatchDetail = lazy(() => import('./components/Matches/MatchDetail'));
const LiveMatch = lazy(() => import('./components/Matches/LiveMatch'));
const MatchPrediction = lazy(() => import('./components/Matches/MatchPrediction'));
const PlayersList = lazy(() => import('./components/Players/PlayersList'));
const PlayerDetail = lazy(() => import('./components/Players/PlayerDetail'));
const TeamsList = lazy(() => import('./components/Teams/TeamsList'));
const TeamDetail = lazy(() => import('./components/Teams/TeamDetail'));
const TeamStats = lazy(() => import('./components/Teams/TeamStats'));
const InjuriesList = lazy(() => import('./components/Injuries/InjuriesList'));
const LeaguesList = lazy(() => import('./components/Leagues/LeaguesList'));
const LeagueDetail = lazy(() => import('./components/Leagues/LeagueDetail'));
const Statistics = lazy(() => import('./pages/Statistics'));
const Profile = lazy(() => import('./pages/Profile'));
const Settings = lazy(() => import('./pages/Settings'));
const Login = lazy(() => import('./pages/Auth/Login'));
const Register = lazy(() => import('./pages/Auth/Register'));
const NotFound = lazy(() => import('./pages/NotFound'));

// =============================================================================
// COMPONENTES DE RUTAS PROTEGIDAS
// =============================================================================
import ProtectedRoute from './components/Auth/ProtectedRoute';
import PublicRoute from './components/Auth/PublicRoute';

// =============================================================================
// ESTILOS
// =============================================================================
import './App.css';

/**
 * Componente principal de la aplicación Football Analytics
 * 
 * Características principales:
 * - Gestión de contextos globales (Auth, App)
 * - Configuración de routing con React Router
 * - Integración de WebSocket para datos en tiempo real
 * - Sistema de temas (claro/oscuro)
 * - Lazy loading para optimización de rendimiento
 * - Error boundaries para manejo robusto de errores
 * - Sistema de notificaciones global
 */
function App() {
  // =============================================================================
  // ESTADO LOCAL
  // =============================================================================
  const [isInitializing, setIsInitializing] = useState(true);
  const [currentTheme, setCurrentTheme] = useState('light');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // =============================================================================
  // HOOKS DE ALMACENAMIENTO LOCAL
  // =============================================================================
  const { value: savedTheme, setValue: setSavedTheme } = useLocalStorage('theme', 'light');
  const { value: sidebarPreference, setValue: setSidebarPreference } = useLocalStorage('sidebar-open', false);

  // =============================================================================
  // INICIALIZACIÓN DE LA APLICACIÓN
  // =============================================================================
  useEffect(() => {
    initializeApp();
  }, []);

  /**
   * Inicializa la aplicación cargando configuraciones y servicios
   */
  const initializeApp = async () => {
    try {
      console.log('🚀 Inicializando Football Analytics...');

      // 1. Cargar tema guardado
      if (savedTheme) {
        setCurrentTheme(savedTheme);
        applyTheme(savedTheme);
      }

      // 2. Cargar preferencia de sidebar
      if (typeof sidebarPreference === 'boolean') {
        setSidebarOpen(sidebarPreference);
      }

      // 3. Verificar autenticación existente
      const isAuthenticated = authService.isAuthenticated();
      if (isAuthenticated) {
        console.log('✅ Usuario autenticado encontrado');
        // El AuthContext manejará la carga del perfil
      }

      // 4. Inicializar WebSocket (solo si está autenticado)
      if (isAuthenticated) {
        console.log('🔌 Conectando WebSocket...');
        webSocketService.connect();
      }

      // 5. Configurar listeners de eventos
      setupEventListeners();

      console.log('✅ Aplicación inicializada correctamente');
    } catch (error) {
      console.error('❌ Error inicializando aplicación:', error);
    } finally {
      setIsInitializing(false);
    }
  };

  /**
   * Configura los event listeners globales
   */
  const setupEventListeners = () => {
    // Listener para cambios de conexión
    window.addEventListener('online', handleConnectionChange);
    window.addEventListener('offline', handleConnectionChange);

    // Listener para cambios de tema del sistema
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', handleSystemThemeChange);

    // Cleanup al desmontar
    return () => {
      window.removeEventListener('online', handleConnectionChange);
      window.removeEventListener('offline', handleConnectionChange);
      mediaQuery.removeEventListener('change', handleSystemThemeChange);
    };
  };

  /**
   * Maneja cambios en la conectividad
   */
  const handleConnectionChange = () => {
    const isOnline = navigator.onLine;
    console.log(`🌐 Estado de conexión: ${isOnline ? 'Online' : 'Offline'}`);
    
    if (isOnline && authService.isAuthenticated()) {
      // Reconectar WebSocket si estamos online y autenticados
      webSocketService.connect();
    } else if (!isOnline) {
      // Manejar estado offline
      console.log('📱 Modo offline activado');
    }
  };

  /**
   * Maneja cambios en el tema del sistema
   */
  const handleSystemThemeChange = (e) => {
    if (savedTheme === 'system') {
      const systemTheme = e.matches ? 'dark' : 'light';
      setCurrentTheme(systemTheme);
      applyTheme(systemTheme);
    }
  };

  /**
   * Aplica el tema al documento
   */
  const applyTheme = (theme) => {
    const root = document.documentElement;
    
    if (theme === 'system') {
      // Detectar preferencia del sistema
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const actualTheme = systemPrefersDark ? 'dark' : 'light';
      root.setAttribute('data-theme', actualTheme);
      setCurrentTheme(actualTheme);
    } else {
      root.setAttribute('data-theme', theme);
      setCurrentTheme(theme);
    }
  };

  /**
   * Cambia el tema de la aplicación
   */
  const handleThemeChange = (newTheme) => {
    setSavedTheme(newTheme);
    applyTheme(newTheme);
  };

  /**
   * Alterna el estado del sidebar
   */
  const toggleSidebar = () => {
    const newState = !sidebarOpen;
    setSidebarOpen(newState);
    setSidebarPreference(newState);
  };

  // =============================================================================
  // COMPONENTE DE CARGA INICIAL
  // =============================================================================
  if (isInitializing) {
    return (
      <div className="App">
        <LoadingScreen 
          message="Inicializando Football Analytics..."
          showProgress={true}
        />
      </div>
    );
  }

  // =============================================================================
  // RENDER PRINCIPAL
  // =============================================================================
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, errorInfo) => {
        console.error('💥 Error capturado por Error Boundary:', error, errorInfo);
      }}
    >
      <AuthProvider>
        <AppProvider>
          <Router>
            <div className="App" data-theme={currentTheme}>
              {/* Header principal */}
              <Header 
                onToggleSidebar={toggleSidebar}
                onThemeChange={handleThemeChange}
                currentTheme={savedTheme}
              />

              <div className="app-container">
                {/* Sidebar de navegación */}
                <Sidebar 
                  isOpen={sidebarOpen}
                  onClose={() => setSidebarOpen(false)}
                />

                {/* Contenido principal */}
                <main className={`app-main ${sidebarOpen ? 'sidebar-open' : ''}`}>
                  <Suspense fallback={<LoadingScreen message="Cargando página..." />}>
                    <Routes>
                      {/* =============================================================================
                          RUTAS PÚBLICAS (Disponibles sin autenticación)
                          ============================================================================= */}
                      <Route 
                        path="/login" 
                        element={
                          <PublicRoute>
                            <Login />
                          </PublicRoute>
                        } 
                      />
                      <Route 
                        path="/register" 
                        element={
                          <PublicRoute>
                            <Register />
                          </PublicRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS PROTEGIDAS (Requieren autenticación)
                          ============================================================================= */}
                      
                      {/* Dashboard principal */}
                      <Route 
                        path="/" 
                        element={
                          <ProtectedRoute>
                            <Dashboard />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/dashboard" 
                        element={<Navigate to="/" replace />}
                      />

                      {/* =============================================================================
                          RUTAS DE PARTIDOS
                          ============================================================================= */}
                      <Route 
                        path="/matches" 
                        element={
                          <ProtectedRoute>
                            <MatchList />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/matches/:id" 
                        element={
                          <ProtectedRoute>
                            <MatchDetail />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/matches/:id/live" 
                        element={
                          <ProtectedRoute>
                            <LiveMatch />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/matches/:id/prediction" 
                        element={
                          <ProtectedRoute>
                            <MatchPrediction />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE JUGADORES
                          ============================================================================= */}
                      <Route 
                        path="/players" 
                        element={
                          <ProtectedRoute>
                            <PlayersList />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/players/:id" 
                        element={
                          <ProtectedRoute>
                            <PlayerDetail />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE EQUIPOS
                          ============================================================================= */}
                      <Route 
                        path="/teams" 
                        element={
                          <ProtectedRoute>
                            <TeamsList />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/teams/:id" 
                        element={
                          <ProtectedRoute>
                            <TeamDetail />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/teams/:id/stats" 
                        element={
                          <ProtectedRoute>
                            <TeamStats />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE LIGAS
                          ============================================================================= */}
                      <Route 
                        path="/leagues" 
                        element={
                          <ProtectedRoute>
                            <LeaguesList />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/leagues/:id" 
                        element={
                          <ProtectedRoute>
                            <LeagueDetail />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE LESIONES
                          ============================================================================= */}
                      <Route 
                        path="/injuries" 
                        element={
                          <ProtectedRoute>
                            <InjuriesList />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE ESTADÍSTICAS Y ANÁLISIS
                          ============================================================================= */}
                      <Route 
                        path="/statistics" 
                        element={
                          <ProtectedRoute>
                            <Statistics />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTAS DE USUARIO
                          ============================================================================= */}
                      <Route 
                        path="/profile" 
                        element={
                          <ProtectedRoute>
                            <Profile />
                          </ProtectedRoute>
                        } 
                      />
                      <Route 
                        path="/settings" 
                        element={
                          <ProtectedRoute>
                            <Settings />
                          </ProtectedRoute>
                        } 
                      />

                      {/* =============================================================================
                          RUTA 404 - NO ENCONTRADO
                          ============================================================================= */}
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </Suspense>
                </main>
              </div>

              {/* Footer */}
              <Footer />

              {/* Sistema de notificaciones */}
              <NotificationContainer />
            </div>
          </Router>
        </AppProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
}

export default App;