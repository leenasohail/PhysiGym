////////
// title: PhyiCell/custom_modules/custom.cpp
//
// language: C++
// date: 2015-2024
// license: BSD-3-Clause
// author: Elmar Bucher, Paul Macklin
// original source code: https://github.com/MathCancer/PhysiCell
// modified source code: https://github.com/elmbeech/physicellembedding
////////


// load library
#include "custom.h"


// constantes variables
//static const double ZERO = 0;
//static const std::vector<double> VECTOR_ZERO (4, ZERO);  // generate a 4 character long vector of zeros.


// functions

void create_cell_types(void) {
    std::cout << "generate cell types ..." << std::endl;
    std::cout << "cell types can only be defined the first episode of the runtime!" << std::endl;

    // put any modifications to default cell definition here if you
    // want to have "inherited" by other cell types.
    // this is a good place to set default functions.

    // cell_default initial definition
    initialize_default_cell_definition();
    cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);

    cell_defaults.functions.volume_update_function = standard_volume_update_function;
    cell_defaults.functions.update_velocity = standard_update_cell_velocity;

    cell_defaults.functions.update_migration_bias = NULL;
    cell_defaults.functions.update_phenotype = NULL;  // update_cell_and_death_parameters_O2_based;
    cell_defaults.functions.custom_cell_rule = NULL;
    cell_defaults.functions.contact_function = NULL;

    cell_defaults.functions.add_cell_basement_membrane_interactions = NULL;
    cell_defaults.functions.calculate_distance_to_membrane = NULL;

    // parse the cell definitions in the XML config file
    initialize_cell_definitions_from_pugixml();

    // generate the maps of cell definitions.
    build_cell_definitions_maps();

    // intializes cell signal and response dictionaries
    setup_signal_behavior_dictionaries();

    // initializ cell rule definitions
    setup_cell_rules();

    // put any modifications to individual cell definitions here.
    // this is a good place to set custom functions.
    cell_defaults.functions.update_phenotype = phenotype_function;
    cell_defaults.functions.custom_cell_rule = custom_function;
    cell_defaults.functions.contact_function = contact_function;

    // summarize the cell defintion setup.
    display_cell_definitions(std::cout);

    return;
}


void setup_microenvironment(void) {
    // set domain parameters

    // put any custom code to set non-homogeneous initial conditions or
    // extra Dirichlet nodes here.

    // initialize BioFVM
    initialize_microenvironment();

    return;
}


void setup_tissue(void) {
    double Xmin = microenvironment.mesh.bounding_box[0];
    double Ymin = microenvironment.mesh.bounding_box[1];
    double Zmin = microenvironment.mesh.bounding_box[2];

    double Xmax = microenvironment.mesh.bounding_box[3];
    double Ymax = microenvironment.mesh.bounding_box[4];
    double Zmax = microenvironment.mesh.bounding_box[5];

    if (default_microenvironment_options.simulate_2D == true) {
        Zmin = 0.0;
        Zmax = 0.0;
    }

    double Xrange = Xmax - Xmin;
    double Yrange = Ymax - Ymin;
    double Zrange = Zmax - Zmin;

    // create some of each type of cell
    Cell* pC;

    for (int k=0; k < cell_definitions_by_index.size(); k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];
        std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl;
        for (int n = 0; n < parameters.ints("number_of_cells"); n++) {
            std::vector<double> position = {0,0,0};
            position[0] = Xmin + UniformRandom() * Xrange;
            position[1] = Ymin + UniformRandom() * Yrange;
            position[2] = Zmin + UniformRandom() * Zrange;

            pC = create_cell(*pCD);
            pC->assign_position(position);
        }
    }
    std::cout << std::endl;

    // load cells from your CSV file (if enabled)
    load_cells_from_pugixml();
    set_parameters_from_distributions();

    // add custom data vector
    //for (int i = 0 ; i < all_cells->size(); i++) {
    //    std::vector<double> vector_double = VECTOR_ZERO;
    //    (*all_cells)[i]->custom_data.add_vector_variable("my_vector", vector_double);
    //}

    return;
}

std::vector<std::string> my_coloring_function(Cell* pCell) {
    return paint_by_number_cell_coloring(pCell);
}

void phenotype_function(Cell* pCell, Phenotype& phenotype, double dt) {
    return;
}

void custom_function(Cell* pCell, Phenotype& phenotype, double dt) {
    return;
}

void contact_function(Cell* pMe, Phenotype& phenoMe, Cell* pOther, Phenotype& phenoOther, double dt) {
    return;
}

//double persistence_time
//the unit is minutes
int set_persistencetime(std::string s_cell_type, double minutes) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.motility.persistence_time = minutes;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Persistence Time: " << pCell->phenotype.motility.persistence_time << std::endl;
    }
    return 0;
}

//double migration_speed
//the unit is micrometers/min
int set_migrationspeed(std::string s_cell_type, double um_minutes) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.motility.migration_speed = um_minutes;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Migration Speed: " << pCell->phenotype.motility.migration_speed << std::endl;
    }
    return 0;
}


//double migration_bias
int set_migrationbias(std::string s_cell_type, double brownian_bias) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.motility.migration_bias = brownian_bias;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Migration Bias: " << pCell->phenotype.motility.migration_bias << std::endl;
    }
    return 0;
}

// double cell_cell_adhesion_strength
int set_cell_cell_adhesion_strength(std::string s_cell_type, double um_minutes) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.mechanics.cell_cell_adhesion_strength = um_minutes;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Cell Cell Adhesion Strength: " << pCell->phenotype.mechanics.cell_cell_adhesion_strength << std::endl;
    }
    return 0;
}

// double cell_cell_repulsion_strength
int set_cell_cell_repulsion_strength(std::string s_cell_type, double um_minutes) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.mechanics.cell_cell_repulsion_strength = um_minutes;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Cell Cell Repulsion Strength: " << pCell->phenotype.mechanics.cell_cell_repulsion_strength << std::endl;
    }
    return 0;
}

// double relative_maximum_adhesion_distance
int set_relative_maximum_adhesion_distance(std::string s_cell_type, double um) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type) {
            pCell->phenotype.mechanics.relative_maximum_adhesion_distance = um;
        }
        //std::cout << "ID: " << pCell->ID << " Type: " << pCell->type_name << " Relative Maxiumm Adhesion Distance: " << pCell->phenotype.mechanics.relative_maximum_adhesion_distance << std::endl;
    }
    return 0;
}

// double cell_adhesion_affinity
int set_cell_adhesion_affinity(std::string s_cell_type_a, std::string s_cell_type_b, double proportion) {
    for (Cell* pCell: (*all_cells)) {
        if (pCell->type_name == s_cell_type_a) {
            pCell->phenotype.mechanics.cell_adhesion_affinity(s_cell_type_b) = proportion;
        }
        //std::cout << "ID: " << pCell->ID << " Type A: " << pCell->type_name << " Type B:" << s_cell_type_b << " Adhesion Affinity: " << pCell->phenotype.mechanics.cell_adhesion_affinity(s_cell_type_b) << std::endl;
    }
    return 0;
}
