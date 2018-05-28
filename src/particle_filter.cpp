/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	//TransferredCoord global_coord
	global_coord.x = 0.0;
	global_coord.y = 0.0;
	// Initialize the index of associated landmark
	default_random_engine gen;

	// create Gaussian distribution for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		// push back each particle to Particle filter
		particles.push_back(particle);
	}
	if (!particles.empty())
	{
		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	bool if_theta_zero = fabs(yaw_rate) <= 0.00001;

	for (vector<Particle>::iterator it = particles.begin(); it != particles.end(); it++)
	{
		if(!if_theta_zero)
		{
			it->x = it->x + (velocity / yaw_rate) * (sin(it->theta + yaw_rate * delta_t) - sin(it->theta));
			it->y = it->y + (velocity / yaw_rate) * (-cos(it->theta + yaw_rate * delta_t) + cos(it->theta));
			it->theta += (yaw_rate * delta_t);
		}
		else
		{
			it->x += (velocity * delta_t) * cos(it->theta);
			it->y += (velocity * delta_t) * sin(it->theta);
		}
		// Adding Gaussian uncertainty
		normal_distribution<double> dist_x(0.0, std_pos[0]);
		normal_distribution<double> dist_y(0.0, std_pos[1]);
		normal_distribution<double> dist_theta(0.0, std_pos[2]);
		it->x += dist_x(gen);
		it->y += dist_y(gen);
		it->theta += dist_theta(gen);
	}
}

particle_associations ParticleFilter::dataAssociation(double std_landmark[], double & sensor_range, 
		const std::vector<LandmarkObs>& observations, const std::vector<Map::single_landmark_s>& landmark_list, 
		Particle & particle) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Initialize weight
	MVGD = 1.0;
	// Particles associations placeholder
	particle_associations par_association;

	double denox = 2.0 * std_landmark[0] * std_landmark[0];
	double denoy = 2.0 * std_landmark[1] * std_landmark[1];
	double term1 = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	// For every of particles, iterate all available observations obtained at this moment
	for (int o = 0; o < observations.size(); o++)
	{
		// Transfer observation from vehicle coordinate to global coordinate
		global_coord.x = observations[o].x * cos(particle.theta) - observations[o].y * sin(particle.theta) + particle.x;
		global_coord.y = observations[o].x * sin(particle.theta) + observations[o].y * cos(particle.theta) + particle.y;
		// Arbitrated optimal distance used to identify the associated landmark
		double best_dist = 1e6;
		// For each observation of every particle, iterate through all landmarks
		for (int l = 0; l < landmark_list.size(); l++)
		{
			double range = hypot(landmark_list[l].x_f - particle.x, landmark_list[l].y_f - particle.y);
			// Sensing range must be within reasonable sensor working range
			if(range <= sensor_range)
			{
				double dist = hypot(landmark_list[l].x_f - global_coord.x, landmark_list[l].y_f - global_coord.y);
				// Pick up the ID of the nearest landmark
				if(dist < best_dist)
				{
					best_dist = dist;
					associated_lm_id = l;
				}
			}
		}
		// Perform Multi-Variant Gaussian Distribution (for radar, two variants should be considered, x and y)
		// for present actual observation and calculated observation of identified landmark and present particle
		// to obtain the weight
		if (associated_lm_id >= 0)
		{
			double xd = global_coord.x - landmark_list[associated_lm_id].x_f;
			double yd = global_coord.y - landmark_list[associated_lm_id].y_f;
			double term2 = ((xd * xd) / denox) + ((yd * yd) / denoy);
			MVGD *= term1 * exp(-term2);
		}
		else
			MVGD *= 1e-6;
		// Pay eatra attention here, landmark ID starts from 1
		par_association.id.push_back(associated_lm_id + 1);
		par_association.x.push_back(global_coord.x);
		par_association.y.push_back(global_coord.y);
	}
	return par_association;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement (look at equation 
	// 3.33
	// http://planning.cs.uiuc.edu/node99.html
	double wt_sum = 0.0;

	for (int i = 0; i < num_particles; i++)
	{
		particle_associations par_association;

		// Perform coordinates transformation and find associations for each observation measurement by iterating each landmark using nearest algorithm
		par_association = dataAssociation(std_landmark, sensor_range, observations, map_landmarks.landmark_list, particles[i]);

		particles[i].weight = MVGD;
		// Set association
		particles[i] = SetAssociations(particles[i], par_association);
		// Accumulate weights for all observation measurement for each particle
		wt_sum += particles[i].weight;
	}
	// Normalizing weights for each particle
	for (int i = 0; i < num_particles; i++)
	{
		particles[i].weight /= wt_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<double> weights;
	for (int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}
	// Create particles placeholder without changing particles quantity
	vector<Particle> particles_holder(num_particles);
	random_device rd;
	default_random_engine generator(rd());

	for (int i = 0; i < num_particles; i++)
	{
		// produces random integers on the interval [0, n), where the probability of each individual integer i is defined as 
		// wi/S, that is the weight of the ith integer divided by the sum of all n weights.
		discrete_distribution<int> index(weights.begin(), weights.end());
		// Pick drawn particle
		particles_holder[i] = particles[index(generator)];
		particles_holder[i].id = i;
	}
	// Map back to the member of particle filter instance
	particles = particles_holder;
}

Particle ParticleFilter::SetAssociations(Particle & particle, const particle_associations & par_association)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates//
    particle.associations = par_association.id;
    particle.sense_x = par_association.x;
    particle.sense_y = par_association.y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
