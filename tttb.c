#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <math.h>

// --- Definisi Konstan dan Struktur ---

#define NN_INPUT_SIZE 18
#define NN_HIDDEN_SIZE 100
#define NN_OUTPUT_SIZE 9
#define LEARNING_RATE 0.1

#define BATCH_SIZE 32
#define MAX_MOVES_PER_GAME 9

typedef struct {
    char board[9];
    int current_player;
} GameState;

typedef struct {
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];
} NeuralNetwork;

typedef struct {
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float outputs[NN_OUTPUT_SIZE];
    int move;
    float reward;
    int num_moves_in_game;
    int move_idx;
} Experience;

// --- Function Prototypes ---
void init_game(GameState *state);

// --- Fungsi-Fungsi Neural Network ---

float relu(float x) { return x > 0 ? x : 0; }
float relu_derivative(float x) { return x > 1e-6f ? 1.0f : 0.0f; }

#define RANDOM_WEIGHT() (((float)rand() / RAND_MAX) - 0.5f)
void init_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN_SIZE; i++) nn->weights_ih[i] = RANDOM_WEIGHT();
    for (int i = 0; i < NN_HIDDEN_SIZE * NN_OUTPUT_SIZE; i++) nn->weights_ho[i] = RANDOM_WEIGHT();
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) nn->biases_h[i] = RANDOM_WEIGHT();
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) nn->biases_o[i] = RANDOM_WEIGHT();
}

void board_to_inputs(const GameState *state, float *inputs) {
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') { inputs[i * 2] = 0; inputs[i * 2 + 1] = 0; }
        else if (state->board[i] == 'X') { inputs[i * 2] = 1; inputs[i * 2 + 1] = 0; }
        else { inputs[i * 2] = 0; inputs[i * 2 + 1] = 1; }
    }
}

void softmax(float *input, float *output, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) if (input[i] > max_val) max_val = input[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        if (sum > 0) output[i] /= sum;
        else output[i] = 1.0f / size;
    }
}

void forward_pass(const NeuralNetwork *nn, const float *inputs, float *hidden, float *outputs) {
    float raw_logits[NN_OUTPUT_SIZE];
    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float sum = nn->biases_h[i];
        for (int j = 0; j < NN_INPUT_SIZE; j++) {
            sum += inputs[j] * nn->weights_ih[j * NN_HIDDEN_SIZE + i];
        }
        hidden[i] = relu(sum);
    }
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        raw_logits[i] = nn->biases_o[i];
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            raw_logits[i] += hidden[j] * nn->weights_ho[j * NN_OUTPUT_SIZE + i];
        }
    }
    softmax(raw_logits, outputs, NN_OUTPUT_SIZE);
}

void backprop(NeuralNetwork *nn, const Experience *exp) {
    float output_deltas[NN_OUTPUT_SIZE];
    float hidden_deltas[NN_HIDDEN_SIZE];

    float target_probs[NN_OUTPUT_SIZE];
    float scaled_reward = exp->reward * (0.5f + 0.5f * (float)exp->move_idx / (float)exp->num_moves_in_game);
    memset(target_probs, 0, NN_OUTPUT_SIZE * sizeof(float));

    if (scaled_reward >= 0) {
        target_probs[exp->move] = 1.0f;
    } else {
        int empty_squares = 0;
        for (int i = 0; i < 9; i++) {
            GameState temp_state;
            init_game(&temp_state);
            for(int j=0; j<exp->move_idx; j++) temp_state.board[exp->move] = (j % 2 == 0) ? 'X' : 'O';
            if (temp_state.board[i] == '.' && i != exp->move) empty_squares++;
        }
        float prob = (empty_squares > 0) ? 1.0f / empty_squares : 0;
        for (int i = 0; i < 9; i++) {
            GameState temp_state;
            init_game(&temp_state);
            for(int j=0; j<exp->move_idx; j++) temp_state.board[exp->move] = (j % 2 == 0) ? 'X' : 'O';
            if (temp_state.board[i] == '.' && i != exp->move) target_probs[i] = prob;
        }
    }
    
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) {
        output_deltas[i] = (exp->outputs[i] - target_probs[i]) * fabsf(scaled_reward);
    }

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        float error = 0;
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            error += output_deltas[j] * nn->weights_ho[i * NN_OUTPUT_SIZE + j];
        }
        hidden_deltas[i] = error * relu_derivative(exp->hidden[i]);
    }

    for (int i = 0; i < NN_HIDDEN_SIZE; i++) {
        for (int j = 0; j < NN_OUTPUT_SIZE; j++) {
            nn->weights_ho[i * NN_OUTPUT_SIZE + j] -= LEARNING_RATE * output_deltas[j] * exp->hidden[i];
        }
    }
    for (int j = 0; j < NN_OUTPUT_SIZE; j++) nn->biases_o[j] -= LEARNING_RATE * output_deltas[j];
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            nn->weights_ih[i * NN_HIDDEN_SIZE + j] -= LEARNING_RATE * hidden_deltas[j] * exp->inputs[i];
        }
    }
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) nn->biases_h[j] -= LEARNING_RATE * hidden_deltas[j];
}

// --- Fungsi-Fungsi Game dan Pelatihan ---

void init_game(GameState *state) { memset(state->board, '.', 9); state->current_player = 0; }
void display_board(const GameState *state) {
    for (int row = 0; row < 3; row++) {
        printf(" %c | %c | %c     %d | %d | %d\n", state->board[row*3], state->board[row*3+1], state->board[row*3+2], row*3, row*3+1, row*3+2);
        if (row < 2) printf("---|---|---    ---|---|---\n");
    }
    printf("\n");
}
int check_game_over(const GameState *state, char *winner) {
    for (int i = 0; i < 3; i++) {
        if (state->board[i*3] != '.' && state->board[i*3] == state->board[i*3+1] && state->board[i*3+1] == state->board[i*3+2]) { *winner = state->board[i*3]; return 1; }
        if (state->board[i] != '.' && state->board[i] == state->board[i+3] && state->board[i+3] == state->board[i+6]) { *winner = state->board[i]; return 1; }
    }
    if (state->board[0] != '.' && state->board[0] == state->board[4] && state->board[4] == state->board[8]) { *winner = state->board[0]; return 1; }
    if (state->board[2] != '.' && state->board[2] == state->board[4] && state->board[4] == state->board[6]) { *winner = state->board[2]; return 1; }
    int empty_tiles = 0;
    for (int i = 0; i < 9; i++) if (state->board[i] == '.') empty_tiles++;
    if (empty_tiles == 0) { *winner = 'T'; return 1; }
    return 0;
}
int get_random_move(const GameState *state) {
    int move;
    do { move = rand() % 9; } while (state->board[move] != '.');
    return move;
}
int get_computer_move(const NeuralNetwork *nn, const GameState *state, int display_probs) {
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float outputs[NN_OUTPUT_SIZE];
    board_to_inputs(state, inputs);
    forward_pass(nn, inputs, hidden, outputs);
    float best_legal_prob = -1.0f;
    int best_move = -1;
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.' && outputs[i] > best_legal_prob) {
            best_move = i;
            best_legal_prob = outputs[i];
        }
    }
    if (display_probs) {
        printf("Probabilitas langkah NN:\n");
        for (int i = 0; i < 9; i++) {
            printf("%5.1f%% ", outputs[i] * 100.0f);
            if ((i + 1) % 3 == 0) printf("\n");
        }
        printf("\n");
    }
    return best_move;
}

char run_game_and_collect_data(const NeuralNetwork *nn, Experience *experiences, int *num_experiences) {
    GameState state;
    char winner = 0;
    int current_game_moves = 0;
    
    init_game(&state);
    
    while (!check_game_over(&state, &winner)) {
        int move;
        if (state.current_player == 0) {
            move = get_random_move(&state);
        } else {
            float inputs[NN_INPUT_SIZE];
            float hidden[NN_HIDDEN_SIZE];
            float outputs[NN_OUTPUT_SIZE];
            board_to_inputs(&state, inputs);
            forward_pass(nn, inputs, hidden, outputs);
            
            experiences[*num_experiences].move = get_computer_move(nn, &state, 0);
            experiences[*num_experiences].move_idx = current_game_moves;
            memcpy(experiences[*num_experiences].inputs, inputs, NN_INPUT_SIZE * sizeof(float));
            memcpy(experiences[*num_experiences].hidden, hidden, NN_HIDDEN_SIZE * sizeof(float));
            memcpy(experiences[*num_experiences].outputs, outputs, NN_OUTPUT_SIZE * sizeof(float));
            (*num_experiences)++;
            
            move = experiences[(*num_experiences) - 1].move;
        }
        
        state.board[move] = (state.current_player == 0) ? 'X' : 'O';
        current_game_moves++;
        state.current_player = !state.current_player;
    }
    
    float reward;
    if (winner == 'T') reward = 0.3f;
    else if (winner == 'O') reward = 1.0f;
    else reward = -2.0f;
    
    for (int i = (*num_experiences) - (current_game_moves / 2); i < *num_experiences; i++) {
        experiences[i].reward = reward;
        experiences[i].num_moves_in_game = current_game_moves;
    }
    return winner;
}

void train_network_with_batch(NeuralNetwork *nn, Experience *batch, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        backprop(nn, &batch[i]);
    }
}

void train_against_random(NeuralNetwork *nn, int num_games) {
    Experience *all_experiences = (Experience*)malloc(num_games * MAX_MOVES_PER_GAME * sizeof(Experience));
    int total_experiences = 0;
    int wins = 0, losses = 0, ties = 0;

    printf("Mengumpulkan data dari %d game acak...\n", num_games);
    for (int i = 0; i < num_games; i++) {
        char winner = run_game_and_collect_data(nn, all_experiences, &total_experiences);
        if (winner == 'O') wins++;
        else if (winner == 'X') losses++;
        else ties++;
    }
    printf("Pengumpulan data selesai. Total pengalaman: %d\n", total_experiences);
    printf("Rasio: Menang: %.1f%%, Kalah: %.1f%%, Seri: %.1f%%\n\n",
           (float)wins * 100 / num_games, (float)losses * 100 / num_games, (float)ties * 100 / num_games);

    printf("Melatih network menggunakan mini-batch...\n");
    int num_batches = total_experiences / BATCH_SIZE;
    for (int i = 0; i < num_batches; i++) {
        train_network_with_batch(nn, &all_experiences[i * BATCH_SIZE], BATCH_SIZE);
    }
    // Latih sisa data jika ada
    if (total_experiences % BATCH_SIZE != 0) {
        train_network_with_batch(nn, &all_experiences[num_batches * BATCH_SIZE], total_experiences % BATCH_SIZE);
    }
    printf("Pelatihan selesai!\n\n");
    free(all_experiences);
}

// --- Main Program ---

void play_game_human_vs_ai(NeuralNetwork *nn, int human_player_is_x) {
    GameState state;
    char winner;
    int num_moves = 0;
    
    init_game(&state);
    state.current_player = human_player_is_x ? 0 : 1;
    
    printf("Selamat datang di Tic Tac Toe! Anda adalah %c, komputer adalah %c.\n",
           human_player_is_x ? 'X' : 'O', human_player_is_x ? 'O' : 'X');
    printf("Masukkan posisi sebagai angka dari 0-8.\n\n");
    
    Experience user_exp[MAX_MOVES_PER_GAME];
    int user_exp_count = 0;

    while (!check_game_over(&state, &winner)) {
        display_board(&state);
        char human_symbol = human_player_is_x ? 'X' : 'O';
        char computer_symbol = human_player_is_x ? 'O' : 'X';
        
        if ((state.current_player == 0 && human_player_is_x) || (state.current_player == 1 && !human_player_is_x)) {
            int move;
            printf("Langkah Anda (0-8): ");
            // Perbaikan ada di sini: menggunakan scanf dengan nilai balik dan menghapus buffer
            if (scanf(" %d", &move) != 1) {
                printf("Input tidak valid! Harap masukkan angka.\n");
                int c;
                while ((c = getchar()) != '\n' && c != EOF); // Menghapus buffer input
                continue;
            }

            if (move < 0 || move > 8 || state.board[move] != '.') {
                printf("Langkah tidak valid! Coba lagi.\n");
                continue;
            }
            state.board[move] = human_symbol;
        } else {
            printf("Langkah komputer:\n");
            // ... (bagian kode komputer sama)
            float inputs[NN_INPUT_SIZE];
            float hidden[NN_HIDDEN_SIZE];
            float outputs[NN_OUTPUT_SIZE];
            board_to_inputs(&state, inputs);
            forward_pass(nn, inputs, hidden, outputs);
            
            Experience exp;
            exp.move = get_computer_move(nn, &state, 1);
            exp.move_idx = num_moves;
            memcpy(exp.inputs, inputs, NN_INPUT_SIZE * sizeof(float));
            memcpy(exp.hidden, hidden, NN_HIDDEN_SIZE * sizeof(float));
            memcpy(exp.outputs, outputs, NN_OUTPUT_SIZE * sizeof(float));
            user_exp[user_exp_count++] = exp;

            state.board[exp.move] = computer_symbol;
            printf("Komputer menempatkan %c di posisi %d\n", computer_symbol, exp.move);
        }
        num_moves++;
        state.current_player = !state.current_player;
    }
    
    display_board(&state);
    if ((winner == 'X' && human_player_is_x) || (winner == 'O' && !human_player_is_x)) printf("Anda menang!\n");
    else if ((winner == 'O' && human_player_is_x) || (winner == 'X' && !human_player_is_x)) printf("Komputer menang!\n");
    else printf("Seri!\n");

    float reward;
    char nn_symbol = !human_player_is_x ? 'X' : 'O';
    if (winner == 'T') reward = 0.3f;
    else if (winner == nn_symbol) reward = 1.0f;
    else reward = -2.0f;
    
    for (int i = 0; i < user_exp_count; i++) {
        user_exp[i].reward = reward;
        user_exp[i].num_moves_in_game = num_moves;
    }
    train_network_with_batch(nn, user_exp, user_exp_count);
}


int main(int argc, char **argv) {
    int random_games = 150000;
    if (argc > 1) random_games = atoi(argv[1]);
    srand(time(NULL));

    NeuralNetwork nn;
    init_neural_network(&nn);

    if (random_games > 0) {
        train_against_random(&nn, random_games);
    } else {
        printf("Melewatkan pelatihan. Network tidak akan secerdas itu.\n");
    }

    while (1) {
        play_game_human_vs_ai(&nn, 1);
        char play_again_choice;
        printf("Main lagi? (y/n): ");
        scanf(" %c", &play_again_choice);
        if (play_again_choice != 'y' && play_again_choice != 'Y') break;
    }
    return 0;
}

