import React from 'react';
import { createRoot } from 'react-dom/client';

// =============================================================================
// ESTILOS GLOBALES
// =============================================================================
import './index.css';

// =============================================================================
// COMPONENTE PRINCIPAL
// =============================================================================
import App from './App';

// =============================================================================
// CONFIGURACIÓN DE PERFORMANCE Y ANALYTICS
// =============================================================================
import reportWebVitals from './reportWebVitals';

// =============================================================================
// CONFIGURACIÓN DE SERVICE WORKER PARA PWA
// =============================================================================
import * as serviceWorkerRegistration from './serviceWorkerRegistration';

// =============================================================================
// SERVICIOS DE ERROR REPORTING
// =============================================================================
import { initializeErrorReporting, logError } from './utils/errorReporting';

// =============================================================================
// CONFIGURACIÓN DE AMBIENTE
// =============================================================================
const isDevelopment = process.env.NODE_ENV === 'development';
const isProduction = process.env.NODE_ENV === 'production';

/**
 * Inicializa los servicios globales de la aplicación
 */
const initializeServices = () => {
  try {
    console.log(`🚀 Iniciando Football Analytics en modo ${process.env.NODE_ENV}`);

    // 1. Configurar error reporting para producción
    if (isProduction) {
      initializeErrorReporting({
        dsn: process.env.REACT_APP_SENTRY_DSN,
        environment: process.env.NODE_ENV,
        release: process.env.REACT_APP_VERSION || '1.0.0'
      });
      console.log('✅ Error reporting inicializado');
    }

    // 2. Configurar analytics si está habilitado
    if (process.env.REACT_APP_ENABLE_ANALYTICS === 'true') {
      initializeAnalytics();
    }

    // 3. Verificar soporte del navegador
    if (!checkBrowserSupport()) {
      showBrowserUnsupportedMessage();
      return false;
    }

    // 4. Configurar listeners globales de performance
    setupPerformanceMonitoring();

    // 5. Configurar interceptor global de errores
    setupGlobalErrorHandlers();

    console.log('✅ Servicios globales inicializados correctamente');
    return true;
  } catch (error) {
    console.error('❌ Error inicializando servicios:', error);
    logError(error, 'Service Initialization Failed');
    return false;
  }
};

/**
 * Inicializa Google Analytics si está configurado
 */
const initializeAnalytics = () => {
  const trackingId = process.env.REACT_APP_GA_TRACKING_ID;
  
  if (!trackingId) {
    console.warn('⚠️ Google Analytics tracking ID no configurado');
    return;
  }

  try {
    // Cargar Google Analytics dinámicamente
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${trackingId}`;
    document.head.appendChild(script);

    // Configurar gtag
    window.dataLayer = window.dataLayer || [];
    function gtag() {
      window.dataLayer.push(arguments);
    }
    window.gtag = gtag;
    
    gtag('js', new Date());
    gtag('config', trackingId, {
      page_title: 'Football Analytics',
      page_location: window.location.href,
      custom_map: {
        'dimension1': 'user_type',
        'dimension2': 'subscription_type'
      }
    });

    console.log('✅ Google Analytics inicializado');
  } catch (error) {
    console.error('❌ Error inicializando Google Analytics:', error);
  }
};

/**
 * Verifica que el navegador soporte las características requeridas
 */
const checkBrowserSupport = () => {
  const requiredFeatures = [
    'Promise',
    'fetch',
    'Map',
    'Set',
    'Symbol',
    'Object.assign',
    'Array.from'
  ];

  const unsupportedFeatures = requiredFeatures.filter(feature => {
    const keys = feature.split('.');
    let obj = window;
    
    for (const key of keys) {
      if (!(key in obj)) return true;
      obj = obj[key];
    }
    
    return false;
  });

  if (unsupportedFeatures.length > 0) {
    console.error('❌ Navegador no soportado. Características faltantes:', unsupportedFeatures);
    return false;
  }

  // Verificar soporte para APIs específicas de la aplicación
  const webAPIs = [
    'localStorage',
    'sessionStorage',
    'WebSocket',
    'Notification'
  ];

  const missingAPIs = webAPIs.filter(api => !(api in window));
  
  if (missingAPIs.length > 0) {
    console.warn('⚠️ Algunas características avanzadas no estarán disponibles:', missingAPIs);
  }

  return true;
};

/**
 * Muestra mensaje de navegador no soportado
 */
const showBrowserUnsupportedMessage = () => {
  const message = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: #1f2937;
      color: white;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      z-index: 9999;
    ">
      <div style="text-align: center; max-width: 500px; padding: 2rem;">
        <h1 style="margin-bottom: 1rem;">⚽ Football Analytics</h1>
        <h2 style="margin-bottom: 2rem; color: #ef4444;">Navegador No Soportado</h2>
        <p style="margin-bottom: 2rem; line-height: 1.6;">
          Tu navegador no soporta todas las características requeridas para Football Analytics.
          Por favor, actualiza a una versión más reciente o usa uno de estos navegadores:
        </p>
        <ul style="list-style: none; padding: 0; margin-bottom: 2rem;">
          <li style="margin: 0.5rem 0;">• Chrome 90+</li>
          <li style="margin: 0.5rem 0;">• Firefox 88+</li>
          <li style="margin: 0.5rem 0;">• Safari 14+</li>
          <li style="margin: 0.5rem 0;">• Edge 90+</li>
        </ul>
        <button 
          onclick="window.location.reload()" 
          style="
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
          "
        >
          Intentar de Nuevo
        </button>
      </div>
    </div>
  `;
  
  document.body.innerHTML = message;
};

/**
 * Configura el monitoreo de performance web vitals
 */
const setupPerformanceMonitoring = () => {
  // Monitorear Web Vitals críticos
  reportWebVitals((metric) => {
    if (isDevelopment) {
      console.log('📊 Web Vital:', metric);
    }

    // Enviar métricas a analytics en producción
    if (isProduction && window.gtag) {
      window.gtag('event', metric.name, {
        event_category: 'Web Vitals',
        event_label: metric.id,
        value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
        non_interaction: true
      });
    }

    // Alertas para métricas críticas en desarrollo
    if (isDevelopment) {
      const thresholds = {
        FCP: 1800,  // First Contentful Paint
        LCP: 2500,  // Largest Contentful Paint
        FID: 100,   // First Input Delay
        CLS: 0.1,   // Cumulative Layout Shift
        TTFB: 800   // Time to First Byte
      };

      if (metric.value > thresholds[metric.name]) {
        console.warn(`⚠️ Performance issue detected: ${metric.name} = ${metric.value}`);
      }
    }
  });

  // Monitorear long tasks (tareas que bloquean el hilo principal)
  if ('PerformanceObserver' in window) {
    try {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.duration > 50) { // Tareas que toman más de 50ms
            console.warn('⚠️ Long task detected:', {
              duration: entry.duration,
              startTime: entry.startTime
            });
          }
        });
      });
      
      observer.observe({ entryTypes: ['longtask'] });
    } catch (error) {
      console.log('ℹ️ Long task monitoring no disponible');
    }
  }
};

/**
 * Configura manejadores globales de errores
 */
const setupGlobalErrorHandlers = () => {
  // Errores de JavaScript no capturados
  window.addEventListener('error', (event) => {
    console.error('💥 Error JavaScript global:', {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: event.error
    });

    if (isProduction) {
      logError(event.error || new Error(event.message), 'Global JavaScript Error', {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      });
    }
  });

  // Promesas rechazadas no manejadas
  window.addEventListener('unhandledrejection', (event) => {
    console.error('💥 Promise rejection no manejada:', event.reason);

    if (isProduction) {
      logError(
        event.reason instanceof Error ? event.reason : new Error(String(event.reason)),
        'Unhandled Promise Rejection'
      );
    }

    // Prevenir que el error aparezca en la consola del navegador
    if (isDevelopment) {
      event.preventDefault();
    }
  });

  // Errores de recursos (imágenes, scripts, etc.)
  window.addEventListener('error', (event) => {
    if (event.target !== window) {
      console.error('💥 Error cargando recurso:', {
        element: event.target.tagName,
        source: event.target.src || event.target.href,
        message: 'Failed to load resource'
      });

      if (isProduction) {
        logError(new Error('Resource Load Error'), 'Resource Load Failed', {
          element: event.target.tagName,
          source: event.target.src || event.target.href
        });
      }
    }
  }, true);
};

/**
 * Optimiza la carga de fuentes para mejor performance
 */
const optimizeFontLoading = () => {
  // Preload de fuentes críticas
  const criticalFonts = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
    'https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap'
  ];

  criticalFonts.forEach(fontUrl => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'style';
    link.href = fontUrl;
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);

    // Cargar la fuente de forma asíncrona
    const styleLink = document.createElement('link');
    styleLink.rel = 'stylesheet';
    styleLink.href = fontUrl;
    styleLink.media = 'print';
    styleLink.onload = function() {
      this.media = 'all';
    };
    document.head.appendChild(styleLink);
  });

  // Fallback para navegadores que no soportan font-display
  if ('FontFace' in window) {
    const interFont = new FontFace('Inter', 'url(https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2)', {
      weight: '400',
      style: 'normal',
      display: 'swap'
    });

    interFont.load().then(() => {
      document.fonts.add(interFont);
    }).catch(error => {
      console.warn('⚠️ Error cargando fuente Inter:', error);
    });
  }
};

/**
 * Función principal de inicialización y renderizado
 */
const initializeApp = async () => {
  try {
    // 1. Optimizar carga de fuentes
    optimizeFontLoading();

    // 2. Inicializar servicios
    const servicesInitialized = initializeServices();
    
    if (!servicesInitialized) {
      console.error('❌ Falló la inicialización de servicios');
      return;
    }

    // 3. Obtener el contenedor raíz
    const container = document.getElementById('root');
    
    if (!container) {
      throw new Error('Elemento raíz no encontrado');
    }

    // 4. Crear la raíz de React 18
    const root = createRoot(container);

    // 5. Configurar el componente raíz con StrictMode en desarrollo
    const RootComponent = isDevelopment ? (
      <React.StrictMode>
        <App />
      </React.StrictMode>
    ) : (
      <App />
    );

    // 6. Renderizar la aplicación
    root.render(RootComponent);

    console.log('✅ Football Analytics renderizado correctamente');

    // 7. Registrar Service Worker para PWA (solo en producción)
    if (isProduction) {
      serviceWorkerRegistration.register({
        onSuccess: (registration) => {
          console.log('✅ Service Worker registrado correctamente');
          
          // Notificar al usuario sobre actualizaciones disponibles
          if (registration.waiting) {
            showUpdateAvailableNotification();
          }
        },
        onUpdate: (registration) => {
          console.log('🔄 Actualización de aplicación disponible');
          showUpdateAvailableNotification(registration);
        }
      });
    } else {
      // En desarrollo, asegurarse de que el SW esté desregistrado
      serviceWorkerRegistration.unregister();
    }

    // 8. Configurar listeners para visibilidad de página
    setupPageVisibilityHandlers();

    // 9. Log de información de la aplicación
    logAppInfo();

  } catch (error) {
    console.error('💥 Error crítico inicializando la aplicación:', error);
    
    if (isProduction) {
      logError(error, 'App Initialization Failed');
    }

    // Mostrar mensaje de error amigable al usuario
    showCriticalErrorMessage(error);
  }
};

/**
 * Muestra notificación de actualización disponible
 */
const showUpdateAvailableNotification = (registration = null) => {
  // Esta función se integrará con el sistema de notificaciones de la app
  console.log('🔄 Nueva versión disponible. Recargar para actualizar.');
  
  // En una implementación completa, aquí se mostraría una notificación
  // usando el NotificationContainer del componente App
  if (window.confirm('Nueva versión de Football Analytics disponible. ¿Recargar ahora?')) {
    if (registration && registration.waiting) {
      registration.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
    window.location.reload();
  }
};

/**
 * Configura manejadores para cambios de visibilidad de página
 */
const setupPageVisibilityHandlers = () => {
  let isPageVisible = !document.hidden;

  const handleVisibilityChange = () => {
    const wasVisible = isPageVisible;
    isPageVisible = !document.hidden;

    if (wasVisible && !isPageVisible) {
      // Página se ocultó
      console.log('📱 Página oculta - pausando operaciones no críticas');
    } else if (!wasVisible && isPageVisible) {
      // Página se mostró
      console.log('📱 Página visible - reanudando operaciones');
    }
  };

  document.addEventListener('visibilitychange', handleVisibilityChange);

  // También escuchar eventos de focus/blur de ventana
  window.addEventListener('focus', () => {
    console.log('🔍 Ventana enfocada');
  });

  window.addEventListener('blur', () => {
    console.log('🔍 Ventana desenfocada');
  });
};

/**
 * Muestra información de la aplicación en la consola
 */
const logAppInfo = () => {
  const appInfo = {
    name: 'Football Analytics',
    version: process.env.REACT_APP_VERSION || '1.0.0',
    environment: process.env.NODE_ENV,
    buildTime: process.env.REACT_APP_BUILD_TIME || 'Unknown',
    commit: process.env.REACT_APP_GIT_COMMIT || 'Unknown',
    browser: navigator.userAgent,
    viewport: `${window.innerWidth}x${window.innerHeight}`,
    language: navigator.language,
    platform: navigator.platform,
    online: navigator.onLine
  };

  console.log('⚽ Football Analytics App Info:', appInfo);

  if (isDevelopment) {
    console.log('🛠️ Modo desarrollo activado');
    console.log('📊 React StrictMode habilitado');
    console.log('🔧 Hot reload disponible');
  }
};

/**
 * Muestra mensaje de error crítico
 */
const showCriticalErrorMessage = (error) => {
  const errorMessage = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: #1f2937;
      color: white;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      z-index: 9999;
    ">
      <div style="text-align: center; max-width: 500px; padding: 2rem;">
        <h1 style="margin-bottom: 1rem;">⚽ Football Analytics</h1>
        <h2 style="margin-bottom: 2rem; color: #ef4444;">Error Crítico</h2>
        <p style="margin-bottom: 2rem; line-height: 1.6;">
          Ha ocurrido un error inesperado al cargar la aplicación.
          Por favor, recarga la página o contacta al soporte técnico.
        </p>
        ${isDevelopment ? `
          <details style="margin-bottom: 2rem; text-align: left;">
            <summary style="cursor: pointer; margin-bottom: 1rem;">Detalles del Error</summary>
            <pre style="
              background: #374151;
              padding: 1rem;
              border-radius: 0.5rem;
              overflow: auto;
              font-size: 0.875rem;
              white-space: pre-wrap;
            ">${error.stack || error.message}</pre>
          </details>
        ` : ''}
        <button 
          onclick="window.location.reload()" 
          style="
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            margin-right: 1rem;
          "
        >
          Recargar Página
        </button>
        <button 
          onclick="window.location.href='/'" 
          style="
            background: #6b7280;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
          "
        >
          Ir al Inicio
        </button>
      </div>
    </div>
  `;
  
  document.body.innerHTML = errorMessage;
};

// =============================================================================
// INICIALIZACIÓN DE LA APLICACIÓN
// =============================================================================
initializeApp();