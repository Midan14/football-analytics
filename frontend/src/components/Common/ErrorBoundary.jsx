import {
    AlertTriangle,
    Bug,
    CheckCircle,
    ChevronDown,
    ChevronUp,
    Copy,
    ExternalLink,
    Home,
    RefreshCw
} from 'lucide-react';
import React from 'react';

// API Configuration for error reporting
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';
const ENABLE_ERROR_REPORTING = process.env.NODE_ENV === 'production';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      isReporting: false,
      reportSent: false,
      copied: false
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { 
      hasError: true,
      error: error
    };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    console.error('Football Analytics Error Boundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // Report error to monitoring service in production
    if (ENABLE_ERROR_REPORTING) {
      this.reportError(error, errorInfo);
    }

    // Track error in analytics
    if (window.gtag) {
      window.gtag('event', 'exception', {
        description: error.toString(),
        fatal: true,
        custom_map: {
          component_stack: errorInfo.componentStack
        }
      });
    }
  }

  reportError = async (error, errorInfo) => {
    if (this.state.isReporting) return;

    this.setState({ isReporting: true });

    try {
      const errorReport = {
        message: error.message,
        stack: error.stack,
        component_stack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        user_agent: navigator.userAgent,
        url: window.location.href,
        user_id: localStorage.getItem('user_id') || 'anonymous',
        app_version: process.env.REACT_APP_VERSION || 'unknown',
        environment: process.env.NODE_ENV
      };

      const response = await fetch(`${API_BASE_URL}/errors/report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorReport)
      });

      if (response.ok) {
        this.setState({ reportSent: true });
      }
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    } finally {
      this.setState({ isReporting: false });
    }
  };

  handleReload = () => {
    // Clear error state and reload
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      reportSent: false
    });

    // Force a full page reload
    window.location.reload();
  };

  handleRetry = () => {
    // Try to recover without full page reload
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false
    });
  };

  handleGoHome = () => {
    // Navigate to home page
    window.location.href = '/';
  };

  toggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails
    }));
  };

  copyErrorDetails = async () => {
    const errorText = `
Football Analytics Error Report
==============================

Error: ${this.state.error?.message || 'Unknown error'}

Stack Trace:
${this.state.error?.stack || 'No stack trace available'}

Component Stack:
${this.state.errorInfo?.componentStack || 'No component stack available'}

Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}
User Agent: ${navigator.userAgent}
App Version: ${process.env.REACT_APP_VERSION || 'unknown'}
    `.trim();

    try {
      await navigator.clipboard.writeText(errorText);
      this.setState({ copied: true });
      setTimeout(() => this.setState({ copied: false }), 2000);
    } catch (err) {
      console.error('Failed to copy error details:', err);
    }
  };

  render() {
    if (this.state.hasError) {
      const { error, errorInfo, showDetails, isReporting, reportSent, copied } = this.state;
      const isDevelopment = process.env.NODE_ENV === 'development';

      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 flex items-center justify-center p-6">
          <div className="max-w-2xl w-full">
            {/* Main Error Card */}
            <div className="bg-white rounded-xl shadow-lg p-8 text-center">
              {/* Error Icon */}
              <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-6">
                <AlertTriangle className="text-red-600" size={32} />
              </div>

              {/* Error Title */}
              <h1 className="text-3xl font-bold text-gray-900 mb-4">
                Oops! Something went wrong
              </h1>

              {/* Error Description */}
              <p className="text-gray-600 text-lg mb-6">
                Football Analytics encountered an unexpected error. Don't worry, our team has been notified 
                and we're working on a fix.
              </p>

              {/* Error Message (Development Only) */}
              {isDevelopment && error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 text-left">
                  <h3 className="font-semibold text-red-800 mb-2">Development Error:</h3>
                  <p className="text-red-700 text-sm font-mono break-all">
                    {error.message}
                  </p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center mb-6">
                <button
                  onClick={this.handleRetry}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
                >
                  <RefreshCw size={16} />
                  Try Again
                </button>
                
                <button
                  onClick={this.handleReload}
                  className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
                >
                  <RefreshCw size={16} />
                  Reload Page
                </button>
                
                <button
                  onClick={this.handleGoHome}
                  className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
                >
                  <Home size={16} />
                  Go Home
                </button>
              </div>

              {/* Error Reporting Status */}
              {ENABLE_ERROR_REPORTING && (
                <div className="mb-6">
                  {isReporting ? (
                    <div className="flex items-center justify-center gap-2 text-blue-600">
                      <RefreshCw className="animate-spin" size={16} />
                      <span>Reporting error...</span>
                    </div>
                  ) : reportSent ? (
                    <div className="flex items-center justify-center gap-2 text-green-600">
                      <CheckCircle size={16} />
                      <span>Error reported successfully</span>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm">
                      Error details have been automatically reported to our team
                    </div>
                  )}
                </div>
              )}

              {/* Technical Details Toggle */}
              <div className="border-t border-gray-200 pt-6">
                <button
                  onClick={this.toggleDetails}
                  className="flex items-center justify-center gap-2 text-gray-600 hover:text-gray-800 transition-colors mx-auto"
                >
                  <Bug size={16} />
                  Technical Details
                  {showDetails ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>

                {/* Error Details */}
                {showDetails && (
                  <div className="mt-4 text-left">
                    <div className="bg-gray-50 rounded-lg p-4 border">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold text-gray-900">Error Information</h4>
                        <button
                          onClick={this.copyErrorDetails}
                          className={`px-3 py-1 text-sm rounded transition-colors flex items-center gap-1 ${
                            copied 
                              ? 'bg-green-100 text-green-700' 
                              : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                          }`}
                        >
                          {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                          {copied ? 'Copied!' : 'Copy'}
                        </button>
                      </div>

                      {/* Error Message */}
                      {error && (
                        <div className="mb-4">
                          <h5 className="font-medium text-gray-700 mb-1">Error Message:</h5>
                          <p className="text-sm font-mono bg-white p-2 rounded border break-all">
                            {error.message}
                          </p>
                        </div>
                      )}

                      {/* Stack Trace (Development Only) */}
                      {isDevelopment && error?.stack && (
                        <div className="mb-4">
                          <h5 className="font-medium text-gray-700 mb-1">Stack Trace:</h5>
                          <pre className="text-xs font-mono bg-white p-2 rounded border overflow-auto max-h-40">
                            {error.stack}
                          </pre>
                        </div>
                      )}

                      {/* Component Stack (Development Only) */}
                      {isDevelopment && errorInfo?.componentStack && (
                        <div className="mb-4">
                          <h5 className="font-medium text-gray-700 mb-1">Component Stack:</h5>
                          <pre className="text-xs font-mono bg-white p-2 rounded border overflow-auto max-h-40">
                            {errorInfo.componentStack}
                          </pre>
                        </div>
                      )}

                      {/* Environment Info */}
                      <div>
                        <h5 className="font-medium text-gray-700 mb-1">Environment:</h5>
                        <div className="text-sm space-y-1">
                          <div><strong>URL:</strong> {window.location.href}</div>
                          <div><strong>Timestamp:</strong> {new Date().toISOString()}</div>
                          <div><strong>App Version:</strong> {process.env.REACT_APP_VERSION || 'unknown'}</div>
                          <div><strong>Environment:</strong> {process.env.NODE_ENV}</div>
                          <div><strong>User Agent:</strong> {navigator.userAgent}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="mt-6 pt-6 border-t border-gray-200">
                <p className="text-gray-500 text-sm">
                  If this problem persists, please contact our support team.
                </p>
                <div className="flex items-center justify-center gap-4 mt-3">
                  <a
                    href="mailto:support@football-analytics.com"
                    className="text-blue-600 hover:text-blue-800 text-sm flex items-center gap-1 transition-colors"
                  >
                    <ExternalLink size={14} />
                    Contact Support
                  </a>
                  <a
                    href="/help"
                    className="text-blue-600 hover:text-blue-800 text-sm flex items-center gap-1 transition-colors"
                  >
                    <ExternalLink size={14} />
                    Help Center
                  </a>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="mt-6 text-center">
              <p className="text-gray-600 text-sm mb-3">
                Quick actions to get back on track:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                <button
                  onClick={() => window.location.href = '/dashboard'}
                  className="px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => window.location.href = '/predictions'}
                  className="px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  Predictions
                </button>
                <button
                  onClick={() => window.location.href = '/analytics'}
                  className="px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  Analytics
                </button>
                <button
                  onClick={() => window.location.href = '/live'}
                  className="px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  Live Matches
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // If no error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;

// Export a HOC version for easy wrapping
export const withErrorBoundary = (Component, errorBoundaryProps = {}) => {
  const WrappedComponent = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
};

// Export error reporting utility
export const reportError = async (error, context = {}) => {
  if (!ENABLE_ERROR_REPORTING) return;

  try {
    const errorReport = {
      message: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString(),
      user_agent: navigator.userAgent,
      url: window.location.href,
      user_id: localStorage.getItem('user_id') || 'anonymous',
      app_version: process.env.REACT_APP_VERSION || 'unknown',
      environment: process.env.NODE_ENV
    };

    await fetch(`${API_BASE_URL}/errors/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(errorReport)
    });
  } catch (reportingError) {
    console.error('Failed to report error:', reportingError);
  }
};