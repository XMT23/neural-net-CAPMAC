/**
 *  main.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funció que entrena la xarxa neuronal definida + Funció que fa el test del
 * model entrenat + programa principal.
 *
 *  @todo: modificar funció d'entrenament per utilitzar epochs i batchs!!!
 *        Epochs, fet (23/01/25)
 *        Batchs implica canviar les funcions de back_propagation i
 * update_weights de training.c, de moment ho deixem com està Revisar
 * print_recognized
 */

#include "main.h"
#include "common/common.h"
#include "initialize/initialize.h"

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

//-----------FREE INPUT------------
void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized(int p, layer Output) {
    int imax = 0;

    for (int i = 1; i < num_out_layer; i++)
        if (Output.actv[i] > Output.actv[imax])
            imax = i;

    if (imax == Validation[p])
        total++;

    if (debug == 2) {
        printf("Patró %d sembla %c i és %d", p, '0' + imax,
               Validation[p]);
        for (int k = 0; k < num_out_layer; k++)
            printf("   %f", Output.actv[k]);
        printf("\n");
    }
}

/**
 * @brief Entrena la xarxa neuronal en base al conjunt d'entrenament
 *
 * @details Primer carrega tots els patrons d'entrenament (loadPatternSet)
 *          Després realitza num_epochs iteracions d'entrenament.
 *          Cada epoch fa:
 *              - Determina aleatòriament l'ordre en que es consideraran els
 * patrons (per evitar overfitting)
 *              - Per cada patró d'entrenament fa el forward_prop (reconeixament
 * del patró pel model actual) i el back_prop i update_weights (ajustament de
 * pesos i biaxos per provar de millorar la precisió del model)
 *
 * @see loadPatternSet, feed_input, forward_prop, back_prop, update_weights,
 * freeInput
 *
 */
void train_neural_net() {
    if (rank == 0) {
        printf("\nTraining...\n");
    }

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int ranpat[num_training_patterns];

    for (int p = 0; p < num_training_patterns; p++)
        ranpat[p] = p;

    // NOTE: Per seguretat, farem aquest barrier
    MPI_Barrier(MPI_COMM_WORLD);
    // Gradient Descent
    struct timeval now;
    gettimeofday(&now, 0);
    for (int it = 0; it < num_epochs; it++) {
        float loss = 0;
        // Train patterns randomly
        for (int p = 0; p < num_training_patterns; p++) {
            int x = rando();
            int np = (x * x) % num_training_patterns;
            int op = ranpat[p];
            ranpat[p] = ranpat[np];
            ranpat[np] = op;
        }

        for (int nb = 0; nb < num_training_patterns/batch_size; nb++) {
            init_deltas();

            // Calculem quina part del batch fa cada procés
            int low = nb * batch_size + rank * batch_size / size;
            int high = nb * batch_size + (rank + 1) * batch_size / size;
            // for (int i = nb*batch_size; i < (nb+1)*batch_size; i++) {
            for (int i = low; i < high; i++) {
                int p = ranpat[i];
                feed_input(p);
                forward_prop();
                loss += compute_loss(p);
                back_prop(p);
            }
            
            for (int l = 0; l < num_layers - 1; l++) {
                int n = (long)num_neurons[l + 1] * num_neurons[l];
                float *buf = malloc(n * sizeof(float));
                MPI_Allreduce(lay[l].dw, buf, n, MPI_FLOAT,
                        MPI_SUM, MPI_COMM_WORLD);
                memcpy(lay[l].dw, buf, n * sizeof(float));
                free(buf);

                buf = malloc(num_neurons[l] * sizeof(float));
                MPI_Allreduce(lay[l].dbias, buf, num_neurons[l], MPI_FLOAT,
                        MPI_SUM, MPI_COMM_WORLD);
                memcpy(lay[l].dbias, buf, num_neurons[l] * sizeof(float));
                free(buf);
            }
            update_weights();
        }

        float loss_total = 0;
        MPI_Reduce(&loss, &loss_total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Epoch %d Loss: %f\n", it, loss_total/num_training_patterns);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    char** rSet;

    if (rank == 0) {
        printf("\nTesting...\n");
    }

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    total = 0;
    for (int i = rank; i < num_test_patterns; i += size) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];

        forward_prop();
        printRecognized(i, lay[num_layers - 1]);
    }

    int total_global = 0;
    MPI_Reduce(&total, &total_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nTotal encerts = %d (/%d) \n", total_global, num_test_patterns);
    }
    freeInput(num_test_patterns, rSet);
}

//-----------MAIN-----------//
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (rank == 0) {
        printf("Number of processes: %d\n", size);
    }

    if (debug == 1)
        printf("argc = %d \n", argc);
    if (argc <= 1)
        readConfiguration("configuration/configfile.txt");
    else
        readConfiguration(argv[1]);

    if (debug == 1)
        printf("FINISH CONFIG \n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    // Start measuring time
    struct timeval begin;
    gettimeofday(&begin, 0);

    // Train
    train_neural_net();

    // Test
    test_nn();

    // Stop measuring time and calculate the elapsed time
    double elapsed = 0.0;
    if (rank == 0) {
        elapsed = elapsedT(begin);
        printf("TRAIN_TIME=%f", elapsed);
    }

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization (rank %d)...\n", rank);

    if (rank == 0) {
        printf("\n\nGoodbye! (%f sec)\n\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
