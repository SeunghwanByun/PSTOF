/*
 * Probabilistic Spatio-Temporal Occupancy Fusion (PSTOF)
 * Research-grade LiDAR perception algorithm in C
 * 
 * Features:
 * - Hierarchical Bayesian Occupancy Grid
 * - Spatio-Temporal Motion Prediction
 * - Multi-Sensor Dempster-Shafer Fusion
 * - Semantic-Aware Clustering
 * - Uncertainty Quantification
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

// ========== CORE DATA STRUCTURES ==========

typedef struct {
    float x, y, z;
} Point3D_t;

typedef struct {
    float data[3][3];
} Matrix3x3_t;

typedef struct {
    float data[9];  // [x, y, z, vx, vy, vz, ax, ay, az]
} StateVector_t;

typedef struct {
    float data[9][9];
} CovarianceMatrix_t;

// Bayesian Voxel with probabilistic occupancy
typedef struct {
    float alpha;                    // Beta distribution parameter
    float beta;                     // Beta distribution parameter
    float hierarchical_prior;       // Prior from higher resolution
    float last_update_time;
    uint32_t evidence_count;
    uint8_t semantic_class;         // 0=unknown, 1=static, 2=dynamic, 3=vehicle, etc.
    float semantic_confidence;
} BayesianVoxel_t;

// Extended Kalman Filter for motion tracking
typedef struct {
    StateVector_t state;                    // [pos, vel, acc]
    CovarianceMatrix_t covariance;         // State uncertainty
    Matrix3x3_t process_noise;             // Q matrix
    Matrix3x3_t measurement_noise;         // R matrix
    float last_prediction_time;
} ExtendedKalmanFilter_t;

// Sensor evidence structure
typedef struct {
    Point3D_t position;
    bool is_occupied;
    float confidence;               // 0.0 - 1.0
    float sensor_reliability;      // 0.0 - 1.0
    uint32_t sensor_id;
    float timestamp;
    uint8_t semantic_hint;          // Semantic information from sensor
} SensorEvidence_t;

// Dempster-Shafer belief structure
typedef struct {
    float mass_static;              // m({static})
    float mass_dynamic;             // m({dynamic}) 
    float mass_unknown;             // m({unknown})
    float mass_ignorance;           // m({static, dynamic, unknown})
} BeliefFunction_t;

// Motion state for prediction
typedef struct {
    Point3D_t position;
    Point3D_t velocity;
    Point3D_t acceleration;
    Matrix3x3_t position_covariance;
    float temporal_consistency;
} MotionState_t;

// Semantic cluster
typedef struct {
    Point3D_t* points;
    uint32_t point_count;
    uint32_t point_capacity;
    uint8_t predicted_class;
    float semantic_confidence;
    
    // Bounding box
    Point3D_t bbox_min, bbox_max;
    
    // Motion tracking
    ExtendedKalmanFilter_t motion_filter;
    MotionState_t current_motion;
    MotionState_t predicted_motions[5];     // Next 5 time steps
    
    // Temporal properties
    float temporal_consistency_score;
    uint32_t age_frames;
    uint32_t cluster_id;
} SemanticCluster_t;

// Hash table for sparse voxel storage
#define HASH_TABLE_SIZE 100003

typedef struct VoxelHashNode {
    uint64_t key;                   // Voxel coordinate hash
    BayesianVoxel_t voxel;
    struct VoxelHashNode* next;
} VoxelHashNode_t;

typedef struct {
    VoxelHashNode_t* buckets[HASH_TABLE_SIZE];
    uint32_t total_voxels;
    float resolution;
    Point3D_t origin;
} SparseVoxelGrid_t;

// Main PSTOF system structure
typedef struct {
    SparseVoxelGrid_t* occupancy_grid;
    SemanticCluster_t* clusters;
    uint32_t cluster_count;
    uint32_t cluster_capacity;
    
    // Gaussian Process for spatial uncertainty
    Point3D_t* gp_training_points;
    float* gp_training_labels;
    uint32_t gp_training_count;
    float* gp_kernel_matrix;
    
    // System parameters
    float resolution;
    float max_range;
    uint32_t max_clusters;
    
    // Performance metrics
    clock_t last_update_time;
    float processing_time_ms;
} PSTOFSystem_t;

// ========== UTILITY FUNCTIONS ==========

// Hash function for voxel coordinates
uint64_t hash_voxel_key(int32_t x, int32_t y, int32_t z) {
    uint64_t key = 0;
    key |= ((uint64_t)(x & 0x1FFFFF)) << 42;    // 21 bits for x
    key |= ((uint64_t)(y & 0x1FFFFF)) << 21;    // 21 bits for y  
    key |= ((uint64_t)(z & 0x1FFFFF));          // 21 bits for z
    return key;
}

// Convert 3D position to voxel key
uint64_t position_to_voxel_key(const Point3D_t* pos, float resolution) {
    int32_t x = (int32_t)(pos->x / resolution);
    int32_t y = (int32_t)(pos->y / resolution);
    int32_t z = (int32_t)(pos->z / resolution);
    return hash_voxel_key(x, y, z);
}

// Hash table operations
uint32_t hash_function(uint64_t key) {
    return (uint32_t)(key % HASH_TABLE_SIZE);
}

BayesianVoxel_t* get_voxel(SparseVoxelGrid_t* grid, uint64_t key) {
    uint32_t bucket = hash_function(key);
    VoxelHashNode_t* node = grid->buckets[bucket];
    
    while (node) {
        if (node->key == key) {
            return &node->voxel;
        }
        node = node->next;
    }
    return NULL;
}

BayesianVoxel_t* create_or_get_voxel(SparseVoxelGrid_t* grid, uint64_t key) {
    BayesianVoxel_t* existing = get_voxel(grid, key);
    if (existing) return existing;
    
    // Create new voxel
    uint32_t bucket = hash_function(key);
    VoxelHashNode_t* new_node = malloc(sizeof(VoxelHashNode_t));
    if (!new_node) return NULL;
    
    new_node->key = key;
    new_node->voxel = (BayesianVoxel_t){
        .alpha = 1.0f,                  // Uniform prior
        .beta = 1.0f,
        .hierarchical_prior = 0.5f,
        .last_update_time = 0.0f,
        .evidence_count = 0,
        .semantic_class = 0,            // Unknown
        .semantic_confidence = 0.0f
    };
    new_node->next = grid->buckets[bucket];
    grid->buckets[bucket] = new_node;
    grid->total_voxels++;
    
    return &new_node->voxel;
}

// Matrix operations
void matrix3x3_multiply(const Matrix3x3_t* a, const Matrix3x3_t* b, Matrix3x3_t* result) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result->data[i][j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}

void matrix3x3_add(const Matrix3x3_t* a, const Matrix3x3_t* b, Matrix3x3_t* result) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
}

// ========== BAYESIAN OCCUPANCY GRID ==========

void update_bayesian_voxel(BayesianVoxel_t* voxel, const SensorEvidence_t* evidence) {
    // Sensor-specific likelihood computation
    float likelihood;
    if (evidence->is_occupied) {
        likelihood = evidence->confidence * evidence->sensor_reliability;
    } else {
        likelihood = (1.0f - evidence->confidence) * evidence->sensor_reliability;
    }
    
    // Hierarchical prior integration
    float effective_prior = 0.7f * (voxel->alpha / (voxel->alpha + voxel->beta)) + 
                           0.3f * voxel->hierarchical_prior;
    
    // Bayesian update with Beta-Binomial conjugate prior
    if (evidence->is_occupied) {
        voxel->alpha += likelihood;
    } else {
        voxel->beta += likelihood;
    }
    
    // Update semantic information
    if (evidence->semantic_hint > 0 && evidence->confidence > 0.7f) {
        // Weighted update of semantic class
        float new_weight = evidence->confidence * evidence->sensor_reliability;
        float old_weight = voxel->semantic_confidence;
        
        if (new_weight > old_weight) {
            voxel->semantic_class = evidence->semantic_hint;
            voxel->semantic_confidence = new_weight;
        }
    }
    
    voxel->evidence_count++;
    voxel->last_update_time = evidence->timestamp;
}

float get_occupancy_probability(const BayesianVoxel_t* voxel) {
    return voxel->alpha / (voxel->alpha + voxel->beta);
}

float get_occupancy_uncertainty(const BayesianVoxel_t* voxel) {
    float sum = voxel->alpha + voxel->beta;
    return (voxel->alpha * voxel->beta) / (sum * sum * (sum + 1.0f));
}

// ========== EXTENDED KALMAN FILTER ==========

void ekf_predict(ExtendedKalmanFilter_t* ekf, float dt) {
    // State transition matrix for constant acceleration model
    // x(k+1) = x(k) + vx*dt + 0.5*ax*dt^2
    // vx(k+1) = vx(k) + ax*dt  
    // ax(k+1) = ax(k)
    
    StateVector_t* state = &ekf->state;
    
    // Position update
    state->data[0] += state->data[3] * dt + 0.5f * state->data[6] * dt * dt;  // x
    state->data[1] += state->data[4] * dt + 0.5f * state->data[7] * dt * dt;  // y
    state->data[2] += state->data[5] * dt + 0.5f * state->data[8] * dt * dt;  // z
    
    // Velocity update
    state->data[3] += state->data[6] * dt;  // vx
    state->data[4] += state->data[7] * dt;  // vy
    state->data[5] += state->data[8] * dt;  // vz
    
    // Acceleration remains constant (could add process noise here)
    
    // Covariance prediction (simplified)
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (i == j) {
                ekf->covariance.data[i][j] += 0.01f * dt;  // Add process noise
            }
        }
    }
    
    ekf->last_prediction_time += dt;
}

void ekf_update(ExtendedKalmanFilter_t* ekf, const Point3D_t* measurement, 
                float measurement_reliability) {
    // Innovation (measurement residual)
    Point3D_t innovation = {
        measurement->x - ekf->state.data[0],
        measurement->y - ekf->state.data[1], 
        measurement->z - ekf->state.data[2]
    };
    
    // Adaptive measurement noise based on reliability
    float adaptive_noise = 1.0f / (measurement_reliability + 0.001f);
    
    // Simplified Kalman gain (assuming measurement matrix is identity for position)
    float gain = ekf->covariance.data[0][0] / (ekf->covariance.data[0][0] + adaptive_noise);
    
    // State update
    ekf->state.data[0] += gain * innovation.x;
    ekf->state.data[1] += gain * innovation.y;
    ekf->state.data[2] += gain * innovation.z;
    
    // Covariance update
    for (int i = 0; i < 3; i++) {
        ekf->covariance.data[i][i] *= (1.0f - gain);
    }
}

MotionState_t get_motion_state(const ExtendedKalmanFilter_t* ekf) {
    MotionState_t motion = {0};
    
    motion.position.x = ekf->state.data[0];
    motion.position.y = ekf->state.data[1];
    motion.position.z = ekf->state.data[2];
    
    motion.velocity.x = ekf->state.data[3];
    motion.velocity.y = ekf->state.data[4];
    motion.velocity.z = ekf->state.data[5];
    
    motion.acceleration.x = ekf->state.data[6];
    motion.acceleration.y = ekf->state.data[7];
    motion.acceleration.z = ekf->state.data[8];
    
    // Extract position covariance
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            motion.position_covariance.data[i][j] = ekf->covariance.data[i][j];
        }
    }
    
    return motion;
}

// ========== DEMPSTER-SHAFER FUSION ==========

BeliefFunction_t create_sensor_belief(const SensorEvidence_t* evidence) {
    BeliefFunction_t belief = {0};
    float confidence = evidence->confidence * evidence->sensor_reliability;
    
    if (evidence->is_occupied) {
        if (evidence->semantic_hint == 1) {         // Static object
            belief.mass_static = confidence;
        } else if (evidence->semantic_hint == 2) {  // Dynamic object
            belief.mass_dynamic = confidence;
        } else {                                    // Unknown semantic
            belief.mass_unknown = confidence;
        }
    } else {
        // Free space evidence
        belief.mass_static = 0.0f;
        belief.mass_dynamic = 0.0f;
    }
    
    // Remaining mass to ignorance
    belief.mass_ignorance = 1.0f - (belief.mass_static + belief.mass_dynamic + belief.mass_unknown);
    belief.mass_ignorance = fmaxf(0.0f, belief.mass_ignorance);
    
    return belief;
}

BeliefFunction_t combine_beliefs(const BeliefFunction_t* belief1, const BeliefFunction_t* belief2) {
    BeliefFunction_t combined = {0};
    
    // Dempster's rule of combination
    float normalization = 0.0f;
    
    // All possible intersections
    combined.mass_static = belief1->mass_static * belief2->mass_static +
                          belief1->mass_static * belief2->mass_ignorance +
                          belief1->mass_ignorance * belief2->mass_static;
    
    combined.mass_dynamic = belief1->mass_dynamic * belief2->mass_dynamic +
                           belief1->mass_dynamic * belief2->mass_ignorance +
                           belief1->mass_ignorance * belief2->mass_dynamic;
    
    combined.mass_unknown = belief1->mass_unknown * belief2->mass_unknown +
                           belief1->mass_unknown * belief2->mass_ignorance +
                           belief1->mass_ignorance * belief2->mass_unknown;
    
    combined.mass_ignorance = belief1->mass_ignorance * belief2->mass_ignorance;
    
    // Calculate normalization (1 - conflict)
    normalization = combined.mass_static + combined.mass_dynamic + 
                   combined.mass_unknown + combined.mass_ignorance;
    
    // Normalize
    if (normalization > 0.001f) {
        combined.mass_static /= normalization;
        combined.mass_dynamic /= normalization;
        combined.mass_unknown /= normalization;
        combined.mass_ignorance /= normalization;
    }
    
    return combined;
}

// ========== SEMANTIC CLUSTERING ==========

float compute_point_distance(const Point3D_t* a, const Point3D_t* b) {
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    float dz = a->z - b->z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

bool should_merge_clusters(const SemanticCluster_t* cluster1, const SemanticCluster_t* cluster2,
                          float spatial_threshold, float semantic_threshold) {
    // Spatial proximity check
    float distance = compute_point_distance(
        &cluster1->current_motion.position,
        &cluster2->current_motion.position
    );
    
    if (distance > spatial_threshold) {
        return false;
    }
    
    // Semantic compatibility check
    if (cluster1->predicted_class != 0 && cluster2->predicted_class != 0) {
        if (cluster1->predicted_class != cluster2->predicted_class) {
            return false;
        }
    }
    
    // Motion consistency check
    Point3D_t vel_diff = {
        cluster1->current_motion.velocity.x - cluster2->current_motion.velocity.x,
        cluster1->current_motion.velocity.y - cluster2->current_motion.velocity.y,
        cluster1->current_motion.velocity.z - cluster2->current_motion.velocity.z
    };
    
    float velocity_difference = sqrtf(vel_diff.x*vel_diff.x + vel_diff.y*vel_diff.y + vel_diff.z*vel_diff.z);
    
    return velocity_difference < 2.0f;  // m/s threshold
}

void add_point_to_cluster(SemanticCluster_t* cluster, const Point3D_t* point) {
    if (cluster->point_count >= cluster->point_capacity) {
        // Resize cluster
        cluster->point_capacity *= 2;
        cluster->points = realloc(cluster->points, 
                                 cluster->point_capacity * sizeof(Point3D_t));
    }
    
    cluster->points[cluster->point_count] = *point;
    cluster->point_count++;
    
    // Update bounding box
    if (cluster->point_count == 1) {
        cluster->bbox_min = *point;
        cluster->bbox_max = *point;
    } else {
        cluster->bbox_min.x = fminf(cluster->bbox_min.x, point->x);
        cluster->bbox_min.y = fminf(cluster->bbox_min.y, point->y);
        cluster->bbox_min.z = fminf(cluster->bbox_min.z, point->z);
        
        cluster->bbox_max.x = fmaxf(cluster->bbox_max.x, point->x);
        cluster->bbox_max.y = fmaxf(cluster->bbox_max.y, point->y);
        cluster->bbox_max.z = fmaxf(cluster->bbox_max.z, point->z);
    }
}

void update_cluster_motion(SemanticCluster_t* cluster, float timestamp) {
    if (cluster->point_count == 0) return;
    
    // Calculate centroid
    Point3D_t centroid = {0};
    for (uint32_t i = 0; i < cluster->point_count; i++) {
        centroid.x += cluster->points[i].x;
        centroid.y += cluster->points[i].y;
        centroid.z += cluster->points[i].z;
    }
    centroid.x /= cluster->point_count;
    centroid.y /= cluster->point_count;  
    centroid.z /= cluster->point_count;
    
    // Update motion filter with centroid
    ekf_update(&cluster->motion_filter, &centroid, 0.8f);
    cluster->current_motion = get_motion_state(&cluster->motion_filter);
    
    // Predict future motions
    ExtendedKalmanFilter_t temp_filter = cluster->motion_filter;
    for (int i = 0; i < 5; i++) {
        ekf_predict(&temp_filter, 0.1f);  // 100ms steps
        cluster->predicted_motions[i] = get_motion_state(&temp_filter);
    }
    
    cluster->age_frames++;
}

// ========== GAUSSIAN PROCESS FOR UNCERTAINTY ==========

float rbf_kernel(const Point3D_t* a, const Point3D_t* b, float length_scale) {
    float distance_sq = (a->x - b->x) * (a->x - b->x) +
                       (a->y - b->y) * (a->y - b->y) +
                       (a->z - b->z) * (a->z - b->z);
    return expf(-distance_sq / (2.0f * length_scale * length_scale));
}

float predict_spatial_uncertainty(PSTOFSystem_t* system, const Point3D_t* query_point) {
    if (system->gp_training_count == 0) return 0.5f;  // Default uncertainty
    
    // Simple GP prediction (simplified implementation)
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (uint32_t i = 0; i < system->gp_training_count; i++) {
        float weight = rbf_kernel(query_point, &system->gp_training_points[i], 1.0f);
        weighted_sum += weight * system->gp_training_labels[i];
        weight_sum += weight;
    }
    
    if (weight_sum > 0.001f) {
        return weighted_sum / weight_sum;
    } else {
        return 0.5f;  // Default uncertainty
    }
}

// ========== MAIN PSTOF SYSTEM ==========

PSTOFSystem_t* create_pstof_system(float resolution, float max_range, uint32_t max_clusters) {
    PSTOFSystem_t* system = malloc(sizeof(PSTOFSystem_t));
    if (!system) return NULL;
    
    // Initialize sparse voxel grid
    system->occupancy_grid = malloc(sizeof(SparseVoxelGrid_t));
    memset(system->occupancy_grid->buckets, 0, sizeof(system->occupancy_grid->buckets));
    system->occupancy_grid->total_voxels = 0;
    system->occupancy_grid->resolution = resolution;
    system->occupancy_grid->origin = (Point3D_t){0, 0, 0};
    
    // Initialize clusters
    system->clusters = malloc(max_clusters * sizeof(SemanticCluster_t));
    system->cluster_count = 0;
    system->cluster_capacity = max_clusters;
    
    // Initialize each cluster
    for (uint32_t i = 0; i < max_clusters; i++) {
        system->clusters[i].points = malloc(100 * sizeof(Point3D_t));
        system->clusters[i].point_count = 0;
        system->clusters[i].point_capacity = 100;
        system->clusters[i].cluster_id = i;
        
        // Initialize motion filter
        memset(&system->clusters[i].motion_filter, 0, sizeof(ExtendedKalmanFilter_t));
        for (int j = 0; j < 9; j++) {
            system->clusters[i].motion_filter.covariance.data[j][j] = 1.0f;
        }
    }
    
    // Initialize GP training data
    system->gp_training_points = malloc(1000 * sizeof(Point3D_t));
    system->gp_training_labels = malloc(1000 * sizeof(float));
    system->gp_training_count = 0;
    
    system->resolution = resolution;
    system->max_range = max_range;
    system->max_clusters = max_clusters;
    
    return system;
}

void destroy_pstof_system(PSTOFSystem_t* system) {
    if (!system) return;
    
    // Free voxel grid
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        VoxelHashNode_t* node = system->occupancy_grid->buckets[i];
        while (node) {
            VoxelHashNode_t* next = node->next;
            free(node);
            node = next;
        }
    }
    free(system->occupancy_grid);
    
    // Free clusters
    for (uint32_t i = 0; i < system->cluster_capacity; i++) {
        free(system->clusters[i].points);
    }
    free(system->clusters);
    
    // Free GP data
    free(system->gp_training_points);
    free(system->gp_training_labels);
    
    free(system);
}

void process_sensor_evidence_batch(PSTOFSystem_t* system, const SensorEvidence_t* evidences, 
                                  uint32_t evidence_count, float timestamp) {
    clock_t start_time = clock();
    
    // Phase 1: Update Bayesian occupancy grid
    for (uint32_t i = 0; i < evidence_count; i++) {
        const SensorEvidence_t* evidence = &evidences[i];
        
        // Skip out-of-range evidence
        float distance = sqrtf(evidence->position.x * evidence->position.x +
                              evidence->position.y * evidence->position.y +
                              evidence->position.z * evidence->position.z);
        if (distance > system->max_range) continue;
        
        uint64_t voxel_key = position_to_voxel_key(&evidence->position, system->resolution);
        BayesianVoxel_t* voxel = create_or_get_voxel(system->occupancy_grid, voxel_key);
        
        if (voxel) {
            update_bayesian_voxel(voxel, evidence);
        }
    }
    
    // Phase 2: Extract dynamic regions and cluster
    system->cluster_count = 0;
    
    // Iterate through all voxels to find dynamic ones
    for (uint32_t bucket = 0; bucket < HASH_TABLE_SIZE; bucket++) {
        VoxelHashNode_t* node = system->occupancy_grid->buckets[bucket];
        
        while (node && system->cluster_count < system->cluster_capacity) {
            BayesianVoxel_t* voxel = &node->voxel;
            
            // Check if voxel indicates dynamic object
            float occupancy_prob = get_occupancy_probability(voxel);
            bool is_dynamic = (occupancy_prob > 0.7f) && 
                             (voxel->semantic_class == 2 || voxel->semantic_class == 0);
            
            if (is_dynamic) {
                // Convert voxel key back to position
                uint64_t key = node->key;
                int32_t vx = (int32_t)((key >> 42) & 0x1FFFFF);
                int32_t vy = (int32_t)((key >> 21) & 0x1FFFFF);
                int32_t vz = (int32_t)(key & 0x1FFFFF);
                
                // Handle signed coordinates
                if (vx & 0x100000) vx |= 0xFFE00000;
                if (vy & 0x100000) vy |= 0xFFE00000;
                if (vz & 0x100000) vz |= 0xFFE00000;
                
                Point3D_t voxel_position = {
                    vx * system->resolution,
                    vy * system->resolution,
                    vz * system->resolution
                };
                
                // Find closest cluster or create new one
                bool added_to_cluster = false;
                for (uint32_t c = 0; c < system->cluster_count; c++) {
                    if (should_merge_clusters(&system->clusters[c], &system->clusters[c], 
                                            2.0f, 0.8f)) {  // Simplified check
                        add_point_to_cluster(&system->clusters[c], &voxel_position);
                        added_to_cluster = true;
                        break;
                    }
                }
                
                if (!added_to_cluster && system->cluster_count < system->cluster_capacity) {
                    // Create new cluster
                    SemanticCluster_t* new_cluster = &system->clusters[system->cluster_count];
                    new_cluster->point_count = 0;
                    new_cluster->predicted_class = voxel->semantic_class;
                    new_cluster->semantic_confidence = voxel->semantic_confidence;
                    add_point_to_cluster(new_cluster, &voxel_position);
                    system->cluster_count++;
                }
            }
            
            node = node->next;
        }
    }
    
    // Phase 3: Update cluster motions and predictions
    for (uint32_t c = 0; c < system->cluster_count; c++) {
        update_cluster_motion(&system->clusters[c], timestamp);
    }
    
    // Phase 4: Update GP training data for uncertainty modeling
    if (system->gp_training_count < 1000) {
        for (uint32_t i = 0; i < evidence_count && system->gp_training_count < 1000; i += 10) {
            const SensorEvidence_t* evidence = &evidences[i];
            system->gp_training_points[system->gp_training_count] = evidence->position;
            system->gp_training_labels[system->gp_training_count] = evidence->confidence;
            system->gp_training_count++;
        }
    }
    
    // Calculate processing time
    clock_t end_time = clock();
    system->processing_time_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;
    system->last_update_time = end_time;
}

// Multi-sensor Dempster-Shafer fusion for a specific voxel
BeliefFunction_t fuse_multi_sensor_evidence(const SensorEvidence_t* evidences, 
                                           uint32_t evidence_count) {
    if (evidence_count == 0) {
        return (BeliefFunction_t){0.0f, 0.0f, 0.0f, 1.0f};  // Pure ignorance
    }
    
    BeliefFunction_t combined_belief = create_sensor_belief(&evidences[0]);
    
    for (uint32_t i = 1; i < evidence_count; i++) {
        BeliefFunction_t sensor_belief = create_sensor_belief(&evidences[i]);
        combined_belief = combine_beliefs(&combined_belief, &sensor_belief);
    }
    
    return combined_belief;
}

// ========== PREDICTION AND UNCERTAINTY QUANTIFICATION ==========

typedef struct {
    Point3D_t position;
    float occupancy_probability;
    float uncertainty;
    uint8_t predicted_class;
    float class_confidence;
} PredictionResult_t;

PredictionResult_t predict_voxel_state(PSTOFSystem_t* system, const Point3D_t* position, 
                                      float future_time) {
    PredictionResult_t result = {0};
    result.position = *position;
    
    // Get current voxel state
    uint64_t voxel_key = position_to_voxel_key(position, system->resolution);
    BayesianVoxel_t* voxel = get_voxel(system->occupancy_grid, voxel_key);
    
    if (voxel) {
        result.occupancy_probability = get_occupancy_probability(voxel);
        result.uncertainty = get_occupancy_uncertainty(voxel);
        result.predicted_class = voxel->semantic_class;
        result.class_confidence = voxel->semantic_confidence;
    } else {
        // No data for this voxel
        result.occupancy_probability = 0.1f;  // Low probability for unknown areas
        result.uncertainty = 0.9f;            // High uncertainty
        result.predicted_class = 0;           // Unknown
        result.class_confidence = 0.0f;
    }
    
    // Incorporate motion predictions from nearby clusters
    for (uint32_t c = 0; c < system->cluster_count; c++) {
        SemanticCluster_t* cluster = &system->clusters[c];
        
        // Check if any predicted position is near our query position
        for (int p = 0; p < 5; p++) {
            float distance = compute_point_distance(position, &cluster->predicted_motions[p].position);
            
            if (distance < system->resolution * 2.0f) {
                // This cluster might affect our prediction
                float influence = expf(-distance / system->resolution);  // Gaussian influence
                
                result.occupancy_probability += influence * 0.8f;  // Increase occupancy
                result.predicted_class = cluster->predicted_class;
                result.class_confidence = fmaxf(result.class_confidence, 
                                               cluster->semantic_confidence * influence);
                break;
            }
        }
    }
    
    // Incorporate spatial uncertainty from GP
    float spatial_uncertainty = predict_spatial_uncertainty(system, position);
    result.uncertainty = fmaxf(result.uncertainty, spatial_uncertainty);
    
    // Clamp probabilities
    result.occupancy_probability = fminf(1.0f, fmaxf(0.0f, result.occupancy_probability));
    
    return result;
}

// ========== ANALYSIS AND DEBUGGING FUNCTIONS ==========

typedef struct {
    uint32_t total_voxels;
    uint32_t occupied_voxels;
    uint32_t dynamic_voxels;
    uint32_t static_voxels;
    uint32_t total_clusters;
    float average_cluster_size;
    float processing_time_ms;
    float memory_usage_mb;
    float uncertainty_mean;
    float uncertainty_std;
} SystemStatistics_t;

SystemStatistics_t analyze_system_performance(PSTOFSystem_t* system) {
    SystemStatistics_t stats = {0};
    
    stats.total_voxels = system->occupancy_grid->total_voxels;
    stats.total_clusters = system->cluster_count;
    stats.processing_time_ms = system->processing_time_ms;
    
    // Analyze voxel occupancy
    float uncertainty_sum = 0.0f;
    float uncertainty_sq_sum = 0.0f;
    
    for (uint32_t bucket = 0; bucket < HASH_TABLE_SIZE; bucket++) {
        VoxelHashNode_t* node = system->occupancy_grid->buckets[bucket];
        
        while (node) {
            BayesianVoxel_t* voxel = &node->voxel;
            float occupancy = get_occupancy_probability(voxel);
            float uncertainty = get_occupancy_uncertainty(voxel);
            
            if (occupancy > 0.5f) {
                stats.occupied_voxels++;
                
                if (voxel->semantic_class == 2) {
                    stats.dynamic_voxels++;
                } else if (voxel->semantic_class == 1) {
                    stats.static_voxels++;
                }
            }
            
            uncertainty_sum += uncertainty;
            uncertainty_sq_sum += uncertainty * uncertainty;
            
            node = node->next;
        }
    }
    
    // Calculate uncertainty statistics
    if (stats.total_voxels > 0) {
        stats.uncertainty_mean = uncertainty_sum / stats.total_voxels;
        float variance = (uncertainty_sq_sum / stats.total_voxels) - 
                        (stats.uncertainty_mean * stats.uncertainty_mean);
        stats.uncertainty_std = sqrtf(fmaxf(0.0f, variance));
    }
    
    // Calculate cluster statistics
    if (stats.total_clusters > 0) {
        uint32_t total_points = 0;
        for (uint32_t c = 0; c < system->cluster_count; c++) {
            total_points += system->clusters[c].point_count;
        }
        stats.average_cluster_size = (float)total_points / stats.total_clusters;
    }
    
    // Estimate memory usage
    stats.memory_usage_mb = (float)(
        sizeof(PSTOFSystem_t) +
        system->occupancy_grid->total_voxels * sizeof(VoxelHashNode_t) +
        system->cluster_capacity * sizeof(SemanticCluster_t) +
        system->gp_training_count * (sizeof(Point3D_t) + sizeof(float))
    ) / (1024.0f * 1024.0f);
    
    return stats;
}

void print_system_statistics(const SystemStatistics_t* stats) {
    printf("=== PSTOF System Performance Statistics ===\n");
    printf("Total Voxels: %u\n", stats->total_voxels);
    printf("Occupied Voxels: %u (%.1f%%)\n", stats->occupied_voxels, 
           100.0f * stats->occupied_voxels / fmaxf(1.0f, stats->total_voxels));
    printf("Dynamic Voxels: %u\n", stats->dynamic_voxels);
    printf("Static Voxels: %u\n", stats->static_voxels);
    printf("Total Clusters: %u\n", stats->total_clusters);
    printf("Average Cluster Size: %.1f points\n", stats->average_cluster_size);
    printf("Processing Time: %.2f ms\n", stats->processing_time_ms);
    printf("Memory Usage: %.2f MB\n", stats->memory_usage_mb);
    printf("Uncertainty Mean: %.3f ± %.3f\n", stats->uncertainty_mean, stats->uncertainty_std);
    printf("==========================================\n");
}

// ========== EXAMPLE USAGE AND TESTING ==========

void generate_synthetic_sensor_data(SensorEvidence_t* evidences, uint32_t count, 
                                   uint32_t sensor_id, float timestamp) {
    srand((unsigned int)time(NULL) + sensor_id);
    
    for (uint32_t i = 0; i < count; i++) {
        evidences[i].position.x = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;  // -50 to 50m
        evidences[i].position.y = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;
        evidences[i].position.z = ((float)rand() / RAND_MAX) * 5.0f;           // 0 to 5m
        
        evidences[i].is_occupied = (rand() % 100) < 30;  // 30% occupancy rate
        evidences[i].confidence = 0.5f + ((float)rand() / RAND_MAX) * 0.5f;   // 0.5-1.0
        evidences[i].sensor_reliability = 0.8f + ((float)rand() / RAND_MAX) * 0.2f;  // 0.8-1.0
        evidences[i].sensor_id = sensor_id;
        evidences[i].timestamp = timestamp;
        
        // Assign semantic hints
        if (evidences[i].is_occupied) {
            int semantic_roll = rand() % 100;
            if (semantic_roll < 40) {
                evidences[i].semantic_hint = 1;  // Static
            } else if (semantic_roll < 70) {
                evidences[i].semantic_hint = 2;  // Dynamic  
            } else {
                evidences[i].semantic_hint = 0;  // Unknown
            }
        } else {
            evidences[i].semantic_hint = 0;  // Free space
        }
    }
}

int main() {
    printf("Initializing PSTOF System...\n");
    
    // Create PSTOF system
    PSTOFSystem_t* system = create_pstof_system(
        0.2f,    // 20cm resolution
        100.0f,  // 100m max range
        1000     // max 1000 clusters
    );
    
    if (!system) {
        printf("Failed to create PSTOF system!\n");
        return -1;
    }
    
    printf("System initialized successfully.\n");
    
    // Simulate multi-sensor LiDAR data processing
    const uint32_t num_sensors = 4;
    const uint32_t points_per_sensor = 10000;
    const uint32_t num_frames = 10;
    
    SensorEvidence_t* sensor_data = malloc(points_per_sensor * sizeof(SensorEvidence_t));
    
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        float timestamp = frame * 0.1f;  // 10Hz processing
        
        printf("\nProcessing frame %u (t=%.1fs)...\n", frame, timestamp);
        
        // Process each sensor
        for (uint32_t sensor = 0; sensor < num_sensors; sensor++) {
            generate_synthetic_sensor_data(sensor_data, points_per_sensor, sensor, timestamp);
            process_sensor_evidence_batch(system, sensor_data, points_per_sensor, timestamp);
        }
        
        // Analyze system performance
        SystemStatistics_t stats = analyze_system_performance(system);
        print_system_statistics(&stats);
        
        // Example prediction
        Point3D_t query_point = {10.0f, 5.0f, 1.0f};
        PredictionResult_t prediction = predict_voxel_state(system, &query_point, timestamp + 0.5f);
        
        printf("Prediction at (%.1f, %.1f, %.1f):\n", 
               query_point.x, query_point.y, query_point.z);
        printf("  Occupancy: %.3f ± %.3f\n", 
               prediction.occupancy_probability, prediction.uncertainty);
        printf("  Class: %u (confidence: %.3f)\n", 
               prediction.predicted_class, prediction.class_confidence);
    }
    
    printf("\nCleaning up...\n");
    free(sensor_data);
    destroy_pstof_system(system);
    
    printf("Demo completed successfully!\n");
    return 0;
}
