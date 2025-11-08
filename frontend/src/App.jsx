/**
 * MedSynth Platform - Main Application Component
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import DatasetUpload from './pages/DatasetUpload';
import Training from './pages/Training';
import TrainingCreate from './pages/TrainingCreate';
import TrainingDetail from './pages/TrainingDetail';
import ImageGallery from './pages/ImageGallery';
import PrivacyMetrics from './pages/PrivacyMetrics';

// Components
import Layout from './components/Layout';
import Loading from './components/Loading';

// Protected Route Component
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <Loading />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

// Public Route Component (redirect if logged in)
function PublicRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <Loading />;
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
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

          {/* Protected Routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />

            <Route path="datasets" element={<Datasets />} />
            <Route path="datasets/upload" element={<DatasetUpload />} />

            <Route path="training" element={<Training />} />
            <Route path="training/create" element={<TrainingCreate />} />
            <Route path="training/:id" element={<TrainingDetail />} />
            <Route path="training/:id/images" element={<ImageGallery />} />
            <Route path="training/:id/privacy" element={<PrivacyMetrics />} />
          </Route>

          {/* Catch all */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App;
