/** @type {import('tailwindcss').Config} */
module.exports = {
  // =============================================================================
  // CONFIGURACIÓN DE CONTENIDO - Dónde buscar clases de Tailwind
  // =============================================================================
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
    // Incluir componentes específicos del proyecto
    "./src/components/**/*.{js,jsx,ts,tsx}",
    "./src/pages/**/*.{js,jsx,ts,tsx}",
    "./src/hooks/**/*.{js,jsx,ts,tsx}",
    "./src/context/**/*.{js,jsx,ts,tsx}"
  ],

  // =============================================================================
  // CONFIGURACIÓN DE MODO OSCURO
  // =============================================================================
  darkMode: 'class', // Habilita modo oscuro con clase 'dark'

  // =============================================================================
  // TEMA PERSONALIZADO PARA FOOTBALL ANALYTICS
  // =============================================================================
  theme: {
    // Extender el tema base de Tailwind con nuestro design system
    extend: {
      // =============================================================================
      // COLORES PERSONALIZADOS DEL PROYECTO
      // =============================================================================
      colors: {
        // Colores primarios (azul)
        primary: {
          50: '#eff6ff',
          100: '#dbeafe', 
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6', // Color principal
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554'
        },

        // Colores secundarios (grises)
        secondary: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280', // Gris principal
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#030712'
        },

        // Estados semánticos
        success: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981', // Verde principal
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22'
        },

        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b', // Amarillo principal
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
          950: '#451a03'
        },

        danger: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444', // Rojo principal
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
          950: '#450a0a'
        },

        // =============================================================================
        // COLORES ESPECÍFICOS DE FÚTBOL
        // =============================================================================
        
        // Posiciones de jugadores
        position: {
          gk: '#fbbf24',    // Amarillo para porteros
          def: '#3b82f6',   // Azul para defensas
          mid: '#10b981',   // Verde para mediocampistas
          fwd: '#ef4444'    // Rojo para delanteros
        },

        // Estados de partidos
        match: {
          live: '#ef4444',      // Rojo para en vivo
          finished: '#10b981',  // Verde para finalizado
          upcoming: '#6b7280',  // Gris para próximo
          postponed: '#f59e0b', // Amarillo para pospuesto
          cancelled: '#9ca3af'  // Gris claro para cancelado
        },

        // Severidad de lesiones
        injury: {
          minor: '#10b981',    // Verde para menor
          moderate: '#f59e0b', // Amarillo para moderada
          major: '#ef4444',    // Rojo para grave
          critical: '#dc2626'  // Rojo oscuro para crítica
        },

        // Niveles de confianza en predicciones
        prediction: {
          low: '#ef4444',     // Baja confianza
          medium: '#f59e0b',  // Media confianza
          high: '#10b981'     // Alta confianza
        },

        // =============================================================================
        // COLORES DE BACKGROUND Y SUPERFICIE
        // =============================================================================
        background: {
          primary: '#ffffff',    // Fondo principal claro
          secondary: '#f8fafc',  // Fondo secundario claro
          tertiary: '#f1f5f9',   // Fondo terciario claro
          card: '#ffffff',       // Fondo de tarjetas
          overlay: 'rgba(0, 0, 0, 0.5)' // Overlay para modales
        },

        // Modo oscuro
        dark: {
          primary: '#111827',    // Fondo principal oscuro
          secondary: '#1f2937',  // Fondo secundario oscuro
          tertiary: '#374151',   // Fondo terciario oscuro
          card: '#1f2937',       // Fondo de tarjetas oscuro
          overlay: 'rgba(0, 0, 0, 0.8)' // Overlay más intenso
        }
      },

      // =============================================================================
      // FUENTES PERSONALIZADAS
      // =============================================================================
      fontFamily: {
        // Fuente principal para texto general
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Oxygen',
          'Ubuntu',
          'Cantarell',
          'sans-serif'
        ],
        
        // Fuente para títulos y headings
        heading: [
          'Poppins',
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'sans-serif'
        ],
        
        // Fuente monospace para números y estadísticas
        mono: [
          'JetBrains Mono',
          'Fira Code',
          'Monaco',
          'Consolas',
          'Courier New',
          'monospace'
        ]
      },

      // =============================================================================
      // TAMAÑOS DE FUENTE RESPONSIVE
      // =============================================================================
      fontSize: {
        // Tamaños con line-height optimizado
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
        '7xl': ['4.5rem', { lineHeight: '1' }],
        '8xl': ['6rem', { lineHeight: '1' }],
        '9xl': ['8rem', { lineHeight: '1' }],

        // Tamaños específicos para el proyecto
        'stat': ['2rem', { lineHeight: '2.25rem', fontWeight: '700' }],
        'score': ['3rem', { lineHeight: '3.25rem', fontWeight: '800' }],
        'display': ['4rem', { lineHeight: '4.25rem', fontWeight: '900' }]
      },

      // =============================================================================
      // ESPACIADO PERSONALIZADO
      // =============================================================================
      spacing: {
        // Espacios adicionales para el layout
        '18': '4.5rem',   // 72px
        '22': '5.5rem',   // 88px
        '26': '6.5rem',   // 104px
        '30': '7.5rem',   // 120px
        '34': '8.5rem',   // 136px
        '38': '9.5rem',   // 152px
        '42': '10.5rem',  // 168px
        '46': '11.5rem',  // 184px
        '50': '12.5rem',  // 200px
        '54': '13.5rem',  // 216px
        '58': '14.5rem',  // 232px
        '62': '15.5rem',  // 248px
        '66': '16.5rem',  // 264px
        '70': '17.5rem',  // 280px
        '74': '18.5rem',  // 296px
        '78': '19.5rem',  // 312px
        '82': '20.5rem',  // 328px
        '86': '21.5rem',  // 344px
        '90': '22.5rem',  // 360px
        '94': '23.5rem',  // 376px
        '98': '24.5rem',  // 392px

        // Espacios específicos del proyecto
        'header': '4rem',          // Altura del header
        'sidebar': '16rem',        // Ancho del sidebar
        'card-padding': '1.5rem',  // Padding de tarjetas
        'section-gap': '3rem'      // Espacio entre secciones
      },

      // =============================================================================
      // BORDES REDONDEADOS
      // =============================================================================
      borderRadius: {
        'xs': '0.125rem',  // 2px
        'sm': '0.25rem',   // 4px
        'md': '0.375rem',  // 6px
        'lg': '0.5rem',    // 8px
        'xl': '0.75rem',   // 12px
        '2xl': '1rem',     // 16px
        '3xl': '1.5rem',   // 24px
        '4xl': '2rem',     // 32px
        'card': '0.75rem', // Para tarjetas del proyecto
        'button': '0.5rem' // Para botones del proyecto
      },

      // =============================================================================
      // SOMBRAS PERSONALIZADAS
      // =============================================================================
      boxShadow: {
        'xs': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'sm': '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
        'md': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
        'xl': '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
        '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
        'inner': 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
        
        // Sombras específicas del proyecto
        'card': '0 4px 6px -1px rgb(0 0 0 / 0.05)',
        'card-hover': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
        'modal': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
        'dropdown': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
        
        // Sombras para modo oscuro
        'dark-card': '0 4px 6px -1px rgb(0 0 0 / 0.3)',
        'dark-modal': '0 25px 50px -12px rgb(0 0 0 / 0.5)'
      },

      // =============================================================================
      // ANIMACIONES Y TRANSICIONES
      // =============================================================================
      animation: {
        // Animaciones personalizadas para el proyecto
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'bounce-soft': 'bounceSoft 0.6s ease-out',
        'pulse-slow': 'pulse 3s infinite',
        'spin-slow': 'spin 3s linear infinite',
        
        // Animaciones específicas de fútbol
        'goal-celebration': 'goalCelebration 1s ease-out',
        'live-pulse': 'livePulse 2s infinite',
        'score-update': 'scoreUpdate 0.5s ease-out'
      },

      // Keyframes para las animaciones
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        slideInRight: {
          '0%': { opacity: '0', transform: 'translateX(20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' }
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' }
        },
        bounceSoft: {
          '0%, 20%, 53%, 80%, 100%': { transform: 'translate3d(0, 0, 0)' },
          '40%, 43%': { transform: 'translate3d(0, -15px, 0)' },
          '70%': { transform: 'translate3d(0, -8px, 0)' },
          '90%': { transform: 'translate3d(0, -2px, 0)' }
        },
        goalCelebration: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.1)' },
          '100%': { transform: 'scale(1)' }
        },
        livePulse: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.02)' }
        },
        scoreUpdate: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.05)' },
          '100%': { transform: 'scale(1)' }
        }
      },

      // =============================================================================
      // BREAKPOINTS PERSONALIZADOS
      // =============================================================================
      screens: {
        'xs': '480px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        
        // Breakpoints específicos del proyecto
        'mobile': { 'max': '640px' },
        'tablet': { 'min': '641px', 'max': '1024px' },
        'desktop': { 'min': '1025px' },
        'wide': { 'min': '1440px' }
      },

      // =============================================================================
      // Z-INDEX LAYERS
      // =============================================================================
      zIndex: {
        'dropdown': '1000',
        'sticky': '1020',
        'fixed': '1030',
        'modal-backdrop': '1040',
        'modal': '1050',
        'popover': '1060',
        'tooltip': '1070',
        'toast': '1080'
      },

      // =============================================================================
      // CONFIGURACIONES ESPECÍFICAS DEL GRID
      // =============================================================================
      gridTemplateColumns: {
        // Grids específicos para el dashboard
        'dashboard': 'repeat(auto-fit, minmax(300px, 1fr))',
        'stats': 'repeat(auto-fit, minmax(200px, 1fr))',
        'matches': 'repeat(auto-fill, minmax(280px, 1fr))',
        'players': 'repeat(auto-fill, minmax(250px, 1fr))',
        'teams': 'repeat(auto-fill, minmax(320px, 1fr))'
      },

      // =============================================================================
      // CONFIGURACIONES DE ASPECTO
      // =============================================================================
      aspectRatio: {
        'card': '4 / 3',
        'banner': '16 / 9',
        'square': '1 / 1',
        'player-photo': '3 / 4',
        'team-logo': '1 / 1'
      }
    }
  },

  // =============================================================================
  // PLUGINS DE TAILWIND
  // =============================================================================
  plugins: [
    // Plugin para forms
    require('@tailwindcss/forms')({
      strategy: 'class', // Solo aplicar a elementos con clase 'form-input', etc.
    }),
    
    // Plugin para typography
    require('@tailwindcss/typography'),
    
    // Plugin para aspect ratio (si usas versión anterior a Tailwind 3.0)
    require('@tailwindcss/aspect-ratio'),

    // Plugin personalizado para utilitades específicas de Football Analytics
    function({ addUtilities, addComponents, theme }) {
      // Utilidades personalizadas
      const newUtilities = {
        // Utilidades para posiciones de jugadores
        '.position-gk': {
          backgroundColor: theme('colors.position.gk'),
          color: '#000'
        },
        '.position-def': {
          backgroundColor: theme('colors.position.def'),
          color: '#fff'
        },
        '.position-mid': {
          backgroundColor: theme('colors.position.mid'),
          color: '#fff'
        },
        '.position-fwd': {
          backgroundColor: theme('colors.position.fwd'),
          color: '#fff'
        },

        // Utilidades para estados de partidos
        '.match-live': {
          borderLeftColor: theme('colors.match.live'),
          borderLeftWidth: '4px',
          animation: 'livePulse 2s infinite'
        },
        '.match-finished': {
          borderLeftColor: theme('colors.match.finished'),
          borderLeftWidth: '4px'
        },
        '.match-upcoming': {
          borderLeftColor: theme('colors.match.upcoming'),
          borderLeftWidth: '4px'
        },

        // Utilidades para texto responsivo
        '.text-responsive-xs': {
          fontSize: 'clamp(0.75rem, 0.7rem + 0.2vw, 0.8rem)'
        },
        '.text-responsive-sm': {
          fontSize: 'clamp(0.875rem, 0.8rem + 0.3vw, 0.95rem)'
        },
        '.text-responsive-base': {
          fontSize: 'clamp(1rem, 0.95rem + 0.3vw, 1.1rem)'
        },
        '.text-responsive-lg': {
          fontSize: 'clamp(1.125rem, 1.05rem + 0.4vw, 1.25rem)'
        },
        '.text-responsive-xl': {
          fontSize: 'clamp(1.25rem, 1.15rem + 0.5vw, 1.4rem)'
        },
        '.text-responsive-2xl': {
          fontSize: 'clamp(1.5rem, 1.35rem + 0.7vw, 1.75rem)'
        }
      };

      // Componentes personalizados
      const newComponents = {
        // Componente de tarjeta base
        '.card': {
          backgroundColor: theme('colors.background.card'),
          borderRadius: theme('borderRadius.card'),
          padding: theme('spacing.card-padding'),
          boxShadow: theme('boxShadow.card'),
          transition: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
          
          '&:hover': {
            boxShadow: theme('boxShadow.card-hover'),
            transform: 'translateY(-2px)'
          }
        },

        // Componente de botón base
        '.btn': {
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: `${theme('spacing.2')} ${theme('spacing.4')}`,
          borderRadius: theme('borderRadius.button'),
          fontSize: theme('fontSize.sm'),
          fontWeight: theme('fontWeight.medium'),
          cursor: 'pointer',
          transition: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
          
          '&:disabled': {
            opacity: '0.6',
            cursor: 'not-allowed'
          }
        },

        // Variantes de botón
        '.btn-primary': {
          backgroundColor: theme('colors.primary.500'),
          color: '#fff',
          
          '&:hover:not(:disabled)': {
            backgroundColor: theme('colors.primary.600')
          }
        },

        '.btn-secondary': {
          backgroundColor: theme('colors.secondary.100'),
          color: theme('colors.secondary.700'),
          
          '&:hover:not(:disabled)': {
            backgroundColor: theme('colors.secondary.200')
          }
        }
      };

      addUtilities(newUtilities);
      addComponents(newComponents);
    }
  ],

  // =============================================================================
  // CONFIGURACIÓN EXPERIMENTAL
  // =============================================================================
  experimental: {
    // Optimizaciones futuras
    optimizeUniversalDefaults: true
  },

  // =============================================================================
  // CONFIGURACIÓN DE FUTURE FLAGS
  // =============================================================================
  future: {
    // Habilitar características futuras de Tailwind
    hoverOnlyWhenSupported: true
  }
};