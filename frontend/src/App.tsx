import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import UploadPage from './pages/UploadPage';
import DataPreviewPage from './pages/DataPreviewPage';
import PreprocessPage from './pages/PreprocessPage';
import TrainPage from './pages/TrainPage';
import MetricsPage from './pages/MetricsPage';
import DashboardPage from './pages/DashboardPage';
import InferencePage from './pages/InferencePage';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/upload" element={<ProtectedRoute><UploadPage /></ProtectedRoute>} />
          <Route path="/preview/:fileId" element={<ProtectedRoute><DataPreviewPage /></ProtectedRoute>} />
          <Route path="/preprocess/:fileId" element={<ProtectedRoute><PreprocessPage /></ProtectedRoute>} />
          <Route path="/train/:fileId" element={<ProtectedRoute><TrainPage /></ProtectedRoute>} />
          <Route path="/metrics/:jobId" element={<ProtectedRoute><MetricsPage /></ProtectedRoute>} />
          <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
          <Route path="/inference/:jobId" element={<ProtectedRoute><InferencePage /></ProtectedRoute>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
