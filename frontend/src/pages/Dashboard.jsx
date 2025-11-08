/**
 * MedSynth Platform - Dashboard Page
 */
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { datasetAPI, trainingAPI } from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState({
    datasets: 0,
    training_jobs: 0,
    synthetic_images: 0,
    recent_jobs: [],
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [datasetsRes, trainingRes] = await Promise.all([
        datasetAPI.getStats(),
        trainingAPI.list({ page: 1, page_size: 5 }),
      ]);

      setStats({
        datasets: datasetsRes.data.total_datasets || 0,
        training_jobs: trainingRes.data.total || 0,
        synthetic_images: 0, // Would calculate from training jobs
        recent_jobs: trainingRes.data.jobs || [],
      });
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      completed: 'bg-green-100 text-green-800',
      running: 'bg-blue-100 text-blue-800',
      pending: 'bg-yellow-100 text-yellow-800',
      queued: 'bg-gray-100 text-gray-800',
      failed: 'bg-red-100 text-red-800',
      cancelled: 'bg-gray-100 text-gray-800',
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-primary-600 to-medical-600 rounded-lg shadow-lg p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Welcome to MedSynth Platform</h2>
        <p className="text-primary-100">
          Generate privacy-preserving synthetic medical images with federated learning
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Datasets</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">{stats.datasets}</p>
            </div>
            <div className="p-3 bg-primary-100 rounded-full">
              <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
            </div>
          </div>
          <Link to="/datasets" className="text-sm text-primary-600 hover:text-primary-700 mt-4 inline-block">
            View all datasets →
          </Link>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Training Jobs</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">{stats.training_jobs}</p>
            </div>
            <div className="p-3 bg-medical-100 rounded-full">
              <svg className="w-8 h-8 text-medical-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
          </div>
          <Link to="/training" className="text-sm text-medical-600 hover:text-medical-700 mt-4 inline-block">
            View all jobs →
          </Link>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Synthetic Images</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">{stats.synthetic_images}</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-4">Generated this month</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link
            to="/datasets/upload"
            className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors"
          >
            <div className="flex items-center">
              <div className="p-2 bg-primary-100 rounded-lg">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="font-medium text-gray-900">Upload Dataset</p>
                <p className="text-sm text-gray-600">Add new medical imaging data</p>
              </div>
            </div>
          </Link>

          <Link
            to="/training/create"
            className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-medical-500 hover:bg-medical-50 transition-colors"
          >
            <div className="flex items-center">
              <div className="p-2 bg-medical-100 rounded-lg">
                <svg className="w-6 h-6 text-medical-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="font-medium text-gray-900">New Training Job</p>
                <p className="text-sm text-gray-600">Start federated GAN training</p>
              </div>
            </div>
          </Link>
        </div>
      </div>

      {/* Recent Training Jobs */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Recent Training Jobs</h3>
          <Link to="/training" className="text-sm text-primary-600 hover:text-primary-700">
            View all
          </Link>
        </div>

        {stats.recent_jobs.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500">No training jobs yet</p>
            <Link to="/training/create" className="btn btn-primary mt-4 inline-block">
              Create your first training job
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {stats.recent_jobs.map((job) => (
              <Link
                key={job.id}
                to={`/training/${job.id}`}
                className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{job.name}</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      {job.num_clients} clients • Round {job.current_round}/{job.num_rounds}
                    </p>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm font-medium text-gray-900">{job.progress_percentage.toFixed(1)}%</p>
                      <div className="w-32 h-2 bg-gray-200 rounded-full mt-1">
                        <div
                          className="h-2 bg-primary-600 rounded-full"
                          style={{ width: `${job.progress_percentage}%` }}
                        ></div>
                      </div>
                    </div>
                    <span className={`badge ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
