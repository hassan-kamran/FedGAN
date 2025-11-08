import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { trainingAPI } from '../services/api';

export default function Training() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      const response = await trainingAPI.list({ page: 1, page_size: 20 });
      setJobs(response.data.jobs);
    } catch (error) {
      console.error('Error loading jobs:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      completed: 'badge-success',
      running: 'badge-info',
      pending: 'badge-warning',
      failed: 'badge-danger',
    };
    return colors[status] || 'badge-secondary';
  };

  if (loading) return <div className="flex justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div></div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Training Jobs</h2>
        <Link to="/training/create" className="btn btn-primary">New Training Job</Link>
      </div>

      {jobs.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500">No training jobs yet</p>
          <Link to="/training/create" className="btn btn-primary mt-4 inline-block">Create Training Job</Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {jobs.map((job) => (
            <Link key={job.id} to={`/training/${job.id}`} className="card hover:shadow-lg transition-shadow">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-semibold text-lg">{job.name}</h3>
                  <p className="text-sm text-gray-600">{job.num_clients} clients • Round {job.current_round}/{job.num_rounds}</p>
                </div>
                <div className="text-right">
                  <span className={`badge ${getStatusColor(job.status)}`}>{job.status}</span>
                  <div className="mt-2 text-sm text-gray-600">{job.progress_percentage.toFixed(1)}%</div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
