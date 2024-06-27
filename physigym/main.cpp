////////
// title: PhyiCell/main.cpp
//
// language: C/C++
// date: 2015-2024
// license: BSD-3-Clause
// author: Paul Macklin
// modification: Elmar Bucher, Alexandre Bertin
// original source code: https://github.com/MathCancer/PhysiCell
// modified source code: https://github.com/elmbeech/physicellembedding
// modified source code: https://github.com/Dante-Berth/PhysiGym
//
// description:
//   for the physicell python embedding the content of this regular main.cpp
//   was ported to the physicellmodule.cpp file.
//   this main.cpp was kept, to be still able to run the model the classic way,
//   although slightly adapted to be compatible with physicellmodule.cpp.
////////


// load standard library
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>

// loade physicell library
#include "./core/PhysiCell.h"
#include "./modules/PhysiCell_standard_modules.h"
#include "./custom_modules/custom.h"

// load namespace
using namespace BioFVM;
using namespace PhysiCell;

// main function
int main(int argc, char* argv[]) {

    ///////////
    // start //
    ///////////

    // load and parse settings file(s)
    bool XML_status = false;
    char copy_command [1024];

    if (argc > 1) {
        XML_status = load_PhysiCell_config_file(argv[1]);
        sprintf(copy_command, "cp %s %s", argv[1], PhysiCell_settings.folder.c_str());
    } else {
        XML_status = load_PhysiCell_config_file("./config/PhysiCell_settings.xml");
        sprintf(copy_command, "cp ./config/PhysiCell_settings.xml %s", PhysiCell_settings.folder.c_str());
    }
    if (!XML_status) { exit(-1); }

    // copy config file to output directry
    system(copy_command);

    // OpenMP setup
    omp_set_num_threads(PhysiCell_settings.omp_num_threads);

    // time setup
    std::string time_units = "min";

    /* Microenvironment setup */

    setup_microenvironment(); // modify this in the custom code

    /* PhysiCell setup */

    // set mechanics voxel size, and match the data structure to BioFVM
    double mechanics_voxel_size = 30;
    Cell_Container* cell_container = create_cell_container_for_microenvironment(microenvironment, mechanics_voxel_size);

    /* Users typically start modifying here. START USERMODS */
    random_seed();
    generate_cell_types();  // bue 20240624: delete cells; (re)load cell definitions
    setup_tissue();
    /* Users typically stop modifying here. END USERMODS */

    // set MultiCellDS save options
    set_save_biofvm_mesh_as_matlab(true);
    set_save_biofvm_data_as_matlab(true);
    set_save_biofvm_cell_data(true);
    set_save_biofvm_cell_data_as_custom_matlab(true);

    // save initial data simulation snapshot
    char filename[1024];
    sprintf(filename, "%s/initial", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    // save data simulation snapshot output00000000
    if (PhysiCell_settings.enable_full_saves == true) {
        sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index);
        save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
    }

    // save initial svg cross section through z = 0 and legend
    PhysiCell_SVG_options.length_bar = 200;  // set cross section length bar to 200 microns
    std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // set a pathology coloring function

    sprintf(filename, "%s/legend.svg", PhysiCell_settings.folder.c_str());
    create_plot_legend(filename, cell_coloring_function);

    sprintf(filename, "%s/initial.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);

    // save svg cross section snapshot00000000
    if (PhysiCell_settings.enable_SVG_saves == true) {
        sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
        SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);
    }

    // save legacy simulation report
    std::ofstream report_file;
    if (PhysiCell_settings.enable_legacy_saves == true) {
        sprintf(filename, "%s/simulation_report.txt", PhysiCell_settings.folder.c_str());
        report_file.open(filename);  // create the data log file
        report_file << "simulated time\tnum cells\tnum division\tnum death\twall time" << std::endl;
        log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);  // output00000000
    }

    // standard output
    display_citations();
    display_simulation_status(std::cout);  // output00000000

    // set the performance timers
    BioFVM::RUNTIME_TIC();
    BioFVM::TIC();


    //////////
    // step //
    //////////

    // main loop
    try {
        // set time variables
        double custom_dt = 60; // min
        double custom_countdown = custom_dt;
        double phenotype_countdown = phenotype_dt;
        double mechanics_countdown = mechanics_dt;
        double mcds_countdown = PhysiCell_settings.full_save_interval;
        double svg_countdown = PhysiCell_settings.SVG_save_interval;

        // run diffusion time step paced main loop
        bool step = true;
        while (step) {

            // max time reached?
            if (PhysiCell_globals.current_time > PhysiCell_settings.max_time) {
                step = false;
            }

            // on custom time step
            if (custom_countdown < diffusion_dt / 3) {
                custom_countdown += custom_dt;

                // Put custom time scale code here!
                //std::cout << "processing custom time step block ... " << std::endl;
                /* Custom add-ons could potentially go here. */
            }

            // on phenotype time step
            if (phenotype_countdown < diffusion_dt / 3) {
                phenotype_countdown += phenotype_dt;

                // Put phenotype time scale code here!
                //std::cout << "processing phenotype time step observation block ... " << std::endl;
            }

            // on mechanics time step
            if (mechanics_countdown < diffusion_dt / 3) {
                mechanics_countdown += mechanics_dt;

                // Put mechanics time scale code here!
                //std::cout << "processing mechanic time step observation block ... " << std::endl;
            }

            // on diffusion time step
            // Put diffusion time scale code here!
            //std::cout << "processing diffusion time step observation block ... " << std::endl << std::endl;

            // run microenvironment
            microenvironment.simulate_diffusion_decay(diffusion_dt);

            // run PhysiCell
            ((Cell_Container *)microenvironment.agent_container)->update_all_cells (PhysiCell_globals.current_time);

            // update time
            custom_countdown -= diffusion_dt;
            phenotype_countdown -= diffusion_dt;
            mechanics_countdown -= diffusion_dt;
            mcds_countdown -= diffusion_dt;
            svg_countdown -= diffusion_dt;
            PhysiCell_globals.current_time += diffusion_dt;

            // save data if it's time.
            if (mcds_countdown < diffusion_dt / 3) {
                mcds_countdown += PhysiCell_settings.full_save_interval;
                PhysiCell_globals.full_output_index++;

                display_simulation_status(std::cout);

                // save data simulation snapshot
                if (PhysiCell_settings.enable_full_saves == true) {
                    sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(), PhysiCell_globals.full_output_index);

                    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
                }

                // save legacy simulation report
                if (PhysiCell_settings.enable_legacy_saves == true) {
                    log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
                }
            }

            // save svg plot if it's time
            if ((PhysiCell_settings.enable_SVG_saves == true) and (svg_countdown < diffusion_dt / 3)) {
                svg_countdown += PhysiCell_settings.SVG_save_interval;
                PhysiCell_globals.SVG_output_index++;

                // save final svg cross section
                sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
                SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);
            }
        }

    } catch (const std::exception& e) {  // reference to the base of a polymorphic object
        std::cout << e.what();  // information from length_error printed
    }

    //////////
    // stop //
    //////////

    // save final data simulation snapshot
    sprintf(filename, "%s/final", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    // save final svg cross section
    sprintf(filename, "%s/final.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function);

    // timer
    std::cout << std::endl << "Total simulation runtime: " << std::endl;
    BioFVM::display_stopwatch_value(std::cout, BioFVM::runtime_stopwatch_value());
    std::cout << std::endl;

    // save legacy simulation report
    if (PhysiCell_settings.enable_legacy_saves == true) {
        log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
        report_file.close();
    }

    // going home
    return 0;
}
