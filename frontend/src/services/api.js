/**
 * MedSynth Platform - API Service
 * Handles all communication with the backend API
 */
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_V1 = `${API_URL}/api/v1`;

// Create axios instance
const api = axios.create({
  baseURL: API_V1,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        const response = await axios.post(`${API_V1}/auth/refresh`, {
          refresh_token: refreshToken,
        });

        const { access_token, refresh_token } = response.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);

        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return api(originalRequest);
      } catch (refreshError) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// Authentication APIs
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  getCurrentUser: () => api.get('/auth/me'),
  changePassword: (data) => api.post('/auth/change-password', data),
};

// Dataset APIs
export const datasetAPI = {
  list: (params) => api.get('/datasets/', { params }),
  create: (data) => api.post('/datasets/', data),
  get: (id) => api.get(`/datasets/${id}`),
  update: (id, data) => api.put(`/datasets/${id}`, data),
  delete: (id) => api.delete(`/datasets/${id}`),
  upload: (id, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/datasets/${id}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  getStats: () => api.get('/datasets/stats'),
};

// Training Job APIs
export const trainingAPI = {
  list: (params) => api.get('/training/', { params }),
  create: (data) => api.post('/training/', data),
  get: (id) => api.get(`/training/${id}`),
  update: (id, data) => api.put(`/training/${id}`, data),
  delete: (id) => api.delete(`/training/${id}`),
  cancel: (id) => api.post(`/training/${id}/cancel`),
  getProgress: (id) => api.get(`/training/${id}/progress`),
  getImages: (id, params) => api.get(`/training/${id}/images`, { params }),
  getPrivacyMetrics: (id) => api.get(`/training/${id}/privacy`),
};

export default api;
