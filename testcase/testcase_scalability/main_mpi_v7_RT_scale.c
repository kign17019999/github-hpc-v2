#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MAX_CITIES 20
#define INFINITE INT_MAX
#define ROOT 0

#define LOOP_1ST 0 ////set 1 to eneble loop all
int START_CITIES=9; //start with 0 to n-1

/* MODE_SEND 0 = send Dist by Bcast       | MODE_SEND 1 = send Dist by Ibcast
   MODE_SEND 2 = send Dist by Send & Recv | MODE_SEND 3 = send Dist by Isend & Irecv */
#define MODE_SEND 0

/* MODE_GATHER 0 = gather by Allgather | MODE_GATHER 1 = gather by Send & Recv */
#define MODE_GATHER 0

#define PRINT_ALL 0 //set 1 to eneble print all
#define SAVE_CSV 1
#define NUM_RESULT 7

//path variable
int n;
int (*dist)[MAX_CITIES], (*best_path)[MAX_CITIES];
int *best_path_bound;

//city initiation variable
int (*init_path)[MAX_CITIES], (*init_visited)[MAX_CITIES];
int *init_bound, *init_path_rank;
int init_position, init_level, init_rank, num_init_path;

//other variable
double (*result)[NUM_RESULT];

#define NUM_BRA_BOU 1 //set 1 to eneble print all
double count_branch_bound;

////////////////
int all_best_bound;
MPI_Request *global_request_Isend;
MPI_Request *global_request_Irecv;
int incoming_bound;
int global_flag;
///////////////

void get_cities_info(char* file_path);
void send_data_to_worker(int rank, int size);
void do_wsp(int rank, int size);
void path_initiation(int *path_i, int path_bound, int *visited_i, int level, int size);
void level_initiation(int size);
void branch_and_bound(int *path, int path_bound, int *visited, int level, int rank, int size);
void gather_result(int rank, int size);
void save_result(int rank, int size, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, char* file_path);
void save_result_csv(double index_time, int rank, char *dist_file, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, double count_bab, double r_best_bound, double r_best_path);
double power(double base, int exponent);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int the_best_path_bound=INFINITE;
    int the_best_start, check_best=0;
    double time_of_the_best;
    int *the_best_path = malloc(sizeof(int[MAX_CITIES]));;

    while(1){
        if(rank==ROOT){
            if(PRINT_ALL==1) printf("====================================================\n");
            if(PRINT_ALL==1) if(LOOP_1ST==1) printf("Start with City %d \n", START_CITIES);
        }

        double start_time1 = MPI_Wtime();

        MPI_Request request1, request2;

        //allocation path variable
        dist = malloc(sizeof(int[MAX_CITIES][MAX_CITIES]));
        best_path = malloc(sizeof(int[size][MAX_CITIES]));
        best_path_bound = malloc(sizeof(int[size]));
        best_path_bound[rank] = INFINITE;

        char* file_path;
        if(argc >=3 && strcmp("-i", argv[1]) == 0){
            char* myArg = argv[2];
            while (myArg[0] == '\'') myArg++;
            while (myArg[strlen(myArg)-1] == '\'') myArg[strlen(myArg)-1] = '\0';;
            file_path = myArg;
        }else{
            char *df_file = "input/dist4";
            if(PRINT_ALL==1) if (rank==ROOT) printf("    [ROOT] The default file (%s) will be used if no input is provided  \n", df_file);
            file_path  = df_file;
        }
        

        if(rank==ROOT){
            get_cities_info(file_path);
            if(PRINT_ALL==1) printf("    [ROOT] number of cities = %d \n", n);
            if(PRINT_ALL==1) printf("    [ROOT] number of processor = %d \n", size);
        }

        //send city data to all processor
        double start_time2 = MPI_Wtime();
        send_data_to_worker(rank, size);
        double end_time2 = MPI_Wtime();

        //solving wsp
        double start_time3 = MPI_Wtime();
        do_wsp(rank, size);
        double end_time3 = MPI_Wtime();

        //gathering result
        double start_time4 = MPI_Wtime();
        gather_result(rank, size);
        double end_time4 = MPI_Wtime();

        if(rank==ROOT){
            int min_dist=INFINITE;
            int index_best_path;
            for(int i=0; i<size; i++){            
                if(best_path_bound[i]<min_dist){
                    index_best_path = i;
                    min_dist = best_path_bound[i];
                }
            }

            if(PRINT_ALL==1) printf("    [ROOT] best of the best is in rank %d, \n", index_best_path);
            if(PRINT_ALL==1) printf("      | best_path: ");
            for(int i = 0; i < n ; i++){
                if(PRINT_ALL==1) printf("%d ", best_path[index_best_path][i]);
            }
            if(PRINT_ALL==1) printf("\n");
            if(PRINT_ALL==1) printf("      | best_path_bound: %d \n", best_path_bound[index_best_path]);

            if(the_best_path_bound>best_path_bound[index_best_path]){
                the_best_path_bound = best_path_bound[index_best_path];
                the_best_start = START_CITIES;
                for(int i=0; i<n; i++) the_best_path[i]=best_path[index_best_path][i];
                check_best=1;
            }

        }

        double end_time1 = MPI_Wtime();

        double total_computing_time = end_time1 - start_time1;
        double sending_time = end_time2 - start_time2;
        double BaB_computing_time = end_time3 - start_time3;
        double gathering_time = end_time4 - start_time4;
        if (rank ==ROOT){
            if(PRINT_ALL==1) printf("    [ROOT] spent total : %f seconds\n", total_computing_time);
            if(PRINT_ALL==1) printf("    [ROOT] spent Send  : %f seconds\n", sending_time);
            if(PRINT_ALL==1) printf("    [ROOT] spent BaB   : %f seconds\n", BaB_computing_time);
            if(PRINT_ALL==1) printf("    [ROOT] spent Gather: %f seconds\n", gathering_time);
        }
        if(check_best==1){
            time_of_the_best = total_computing_time;
            check_best=0;
        }

        if(SAVE_CSV==1) save_result(rank, size, total_computing_time, sending_time, BaB_computing_time, gathering_time, file_path);
     
        free(dist);
        free(best_path);
        free(best_path_bound);
        free(init_path);
        free(init_bound);
        free(init_visited);
        free(init_path_rank);
        
        if(LOOP_1ST==1){
            START_CITIES++;
            if(START_CITIES==n) break;
        }else break;
    }

    if(rank==ROOT){
        printf("best path bound = %d | total time = %f s | path = ", the_best_path_bound, time_of_the_best);
        for(int i=0; i<n; i++) printf("%d ", the_best_path[i]);
        printf("| RT = TRUE");
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}

void get_cities_info(char* file_path) {
    FILE* file = fopen(file_path, "r");
    char line[256];
    fscanf(file, "%d", &n); 

    int row=1;
    int col=0;
    dist[row-1][col]=0;

    while (fgets(line, sizeof(line), file)) {
        col=0;
        char *token = strtok(line, " ");
        while (token != NULL) {
            if(atoi(token)>0){
                dist[row-1][col] = atoi(token);
                dist[col][row-1] = atoi(token);
            }
            token = strtok(NULL, " ");
            col++;
        }
        dist[row][col]=0;
        row++;
    }
    fclose(file);
}

void send_data_to_worker(int rank, int size){
    MPI_Request request1, request2;
    if(rank==ROOT){
        if(MODE_SEND==0){
            // mode 0 = send Dist by Bcast
            if(PRINT_ALL==1) printf("    [ROOT] MODE_SEND = 0 (send Dist by Bcast) \n");
            MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
            MPI_Bcast(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, MPI_COMM_WORLD);

        }else if(MODE_SEND==1){
            // mode 1 = send Dist by Ibcast
            if(PRINT_ALL==1) printf("    [ROOT] MODE_SEND = 1 (send Dist by Ibcast) \n");
            MPI_Ibcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD, &request1);
            MPI_Ibcast(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, MPI_COMM_WORLD, &request2);

        }else if(MODE_SEND==2){
            // mode 2 = send Dist by Send & Recv
            if(PRINT_ALL==1) printf("    [ROOT] MODE_SEND = 2 (send Dist by Send & Recv) \n");
            for (int i = 1; i < size; i++) {
                MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(dist, MAX_CITIES*MAX_CITIES, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

        }else if(MODE_SEND==3){
            // mode 3 = send Dist by Isend & Irecv
            if(PRINT_ALL==1) printf("    [ROOT] MODE_SEND = 3 (send Dist by Isend & Irecv) \n");
            for (int i = 1; i < size; i++) {
                    MPI_Isend(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request1);
                    MPI_Isend(dist, MAX_CITIES*MAX_CITIES, MPI_INT, i, 0, MPI_COMM_WORLD, &request2);
            }
        }
    }else {
        if(MODE_SEND==0){
            // mode 0 = send Dist by Bcast
            MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
            MPI_Bcast(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, MPI_COMM_WORLD);
        }else if(MODE_SEND==1){
            // mode 1 = send Dist by Ibcast
            int flag;
            MPI_Ibcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD, &request1);
            MPI_Ibcast(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, MPI_COMM_WORLD, &request2);
            do {
                MPI_Test(&request1, &flag, MPI_STATUS_IGNORE);
            } while (!flag);
            
            do {
                MPI_Test(&request2, &flag, MPI_STATUS_IGNORE);
            } while (!flag);
        
        }else if(MODE_SEND==2){
            // mode 2 = send Dist by Send & Recv
            MPI_Recv(&n, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }else if(MODE_SEND==3){
            // mode 3 = send Dist by Isend & Irecv
            MPI_Irecv(&n, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &request1);
            MPI_Irecv(dist, MAX_CITIES*MAX_CITIES, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &request2);
            MPI_Wait(&request1, MPI_STATUS_IGNORE);
            MPI_Wait(&request2, MPI_STATUS_IGNORE);

        }
    }
}

void do_wsp(int rank, int size){
    //set initial value
    init_position = init_level = init_rank = count_branch_bound = 0;
    num_init_path=1;
    if(START_CITIES>=n) START_CITIES=0;

    //path and visit for initiation
    int *path_i = malloc(MAX_CITIES * sizeof(int));
    int *visited_i = malloc(MAX_CITIES * sizeof(int));
    for(int i=0; i<MAX_CITIES; i++) visited_i[i]=0;
    path_i[0] = START_CITIES;
    visited_i[START_CITIES] = 1;
    level_initiation(size);

    //path initiation
    init_path=malloc(sizeof(int[num_init_path][MAX_CITIES]));
    init_bound=malloc(num_init_path * sizeof(int));
    init_visited=malloc(sizeof(int[num_init_path][MAX_CITIES]));
    init_path_rank=malloc(num_init_path * sizeof(int));
    path_initiation(path_i, 0, visited_i, 1, size);    

    //////////////////////////////////////
    all_best_bound = INFINITE;
    global_request_Isend = malloc(size * sizeof(MPI_Request));
    global_request_Irecv = malloc(size * sizeof(MPI_Request));
    for(int i=0; i<size; i++){
        if(i!=rank){
            MPI_Isend(&all_best_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Isend[i]);
            MPI_Irecv(&incoming_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Irecv[i]);
        }
    }

    //////////////////////////////////////

    //path and visit for branch_and_bound
    int *path = malloc(MAX_CITIES * sizeof(int));
    int *visited = malloc(MAX_CITIES * sizeof(int));
    for(int i=0; i<num_init_path; i++){
        if(init_path_rank[i]==rank){
            for(int j=0; j<n; j++){
                path[j] = init_path[i][j];
                visited[j] = init_visited[i][j];
            }
            branch_and_bound(path, init_bound[i], visited, init_level, rank, size);        
        }
        
    }

    free(path_i);
    free(visited_i);
    free(path);
    free(visited);
    free(global_request_Irecv);
    free(global_request_Isend);

}

void level_initiation(int size){
    for(int level=1; level<n; level++){
        num_init_path = num_init_path*(n-level);
        if(num_init_path >= size || level==n-1){
            if(level==n-1) init_level=level;
            else init_level=level+1;
            break;
        }
    }
}

void path_initiation(int *path_i, int path_bound, int *visited_i, int level, int size) {
    if (level == init_level) {
        for(int i=0; i<n; i++){
            init_path[init_position][i] = path_i[i];
            init_visited[init_position][i] = visited_i[i];
        }
        init_bound[init_position]=path_bound;
        init_path_rank[init_position]=init_rank;
        init_position++;

        if(init_rank==size-1) init_rank=0;
        else init_rank++;
    } else {
        for (int i = 0; i < n; i++) {
            if (!visited_i[i]) {
                path_i[level] = i;
                visited_i[i] = 1;
                int new_bound = path_bound + dist[i][path_i[level - 1]];
                path_initiation(path_i, new_bound, visited_i, level + 1, size);
                visited_i[i] = 0;
            }
        }
    }
}

void branch_and_bound(int *path, int path_bound, int *visited, int level, int rank, int size) {
    if(NUM_BRA_BOU==1) count_branch_bound+=1;
    if (level == n) {
        for(int i=0; i<size;i++){
            if(i!=rank){
                global_flag=0;
                MPI_Test(&global_request_Irecv[i], &global_flag, MPI_STATUS_IGNORE);
                if(global_flag){
                    //printf("[GETT] %d << %d @ %d | %d", rank, i, incoming_bound, all_best_bound);
                    if(incoming_bound < all_best_bound){
                      //printf(" === SAVE\n");
                      MPI_Cancel(&global_request_Isend[i]);
                      all_best_bound = incoming_bound;
                    }else{
                      //printf("\n");
                    }
                    MPI_Irecv(&incoming_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Irecv[i]);
                }
            }
        }
        if (path_bound < all_best_bound) {
            best_path_bound[rank] = path_bound;
            all_best_bound = path_bound;
            for(int i = 0; i < n; i++) best_path[rank][i] = path[i]+1;
            for(int i = 0; i < size; i++) {
                if(i != rank) {
                    MPI_Cancel(&global_request_Isend[i]);
                    MPI_Isend(&all_best_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Isend[i]);
                    //printf("[SEND] %d @ %d >> %d\n", rank, path_bound, i);
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                path[level] = i;
                visited[i] = 1;
                int new_bound = path_bound + dist[i][path[level - 1]];
            if (new_bound < all_best_bound) branch_and_bound(path, new_bound, visited, level + 1, rank, size);
            visited[i] = 0;
	        }
        }
    }
}

void gather_result(int rank, int size){
    int send_buf_bound = best_path_bound[rank];
    int row_to_gather[MAX_CITIES];
    for (int i = 0; i < MAX_CITIES; i++) {
        row_to_gather[i] = best_path[rank][i];
    }

    if(MODE_GATHER==0){
        // MODE_GATHER 0 = gather by Allgather
        if(rank==ROOT) if(PRINT_ALL==1) printf("    [ROOT] MODE_GATHER = 0 (gather by Allgather) \n");
        MPI_Allgather(&send_buf_bound, 1, MPI_INT, best_path_bound, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&row_to_gather, MAX_CITIES, MPI_INT, best_path, MAX_CITIES, MPI_INT, MPI_COMM_WORLD);
    }else if(MODE_GATHER==1){
        // MODE_GATHER 1 = gather by Send & Recv
        if(rank==ROOT){
            if(PRINT_ALL==1) printf("    [ROOT] MODE_GATHER = 1 (gather by Send & Recv) \n");
            for(int i=1; i<size; i++){
                MPI_Recv(&best_path_bound[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for(int i=1; i<size; i++){
                MPI_Recv(&row_to_gather, MAX_CITIES, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for(int j=0; j<n; j++) best_path[i][j]=row_to_gather[j];
            }
        }else{
            MPI_Send(&send_buf_bound, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
            MPI_Send(&row_to_gather, MAX_CITIES, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
        }
    }    
}

void save_result(int rank, int size, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, char* file_path){
    
    result = malloc(sizeof(double[size][NUM_RESULT]));
    result[rank][0] = total_computing_time;
    result[rank][1] = sending_time;
    result[rank][2] = BaB_computing_time;
    result[rank][3] = gathering_time;
    result[rank][4] = count_branch_bound;
    result[rank][5] = best_path_bound[rank];
    double double_path = power(10, 2*n)*404;
    for(int i=0; i<n; i++){
        double_path+=power(10, (n-i-1)*2)*best_path[rank][i];
    }
    result[rank][6] = double_path;
    
    double row_to_gather_result[NUM_RESULT];
    for (int i = 0; i < NUM_RESULT; i++) row_to_gather_result[i] = result[rank][i];

    MPI_Allgather(row_to_gather_result, NUM_RESULT, MPI_DOUBLE  , result, NUM_RESULT, MPI_DOUBLE  , MPI_COMM_WORLD);

    if(rank==ROOT){
        double index_time = MPI_Wtime();
        for(int i=0; i<size;i++) save_result_csv(index_time, i, file_path, result[i][0], result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]);
    }

    free(result);
}

void save_result_csv(double index_time, int rank, char *dist_file, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, double count_bab, double r_best_bound, double r_best_path) {
    FILE *file;
    char date[20];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    char* fileName="result_parallel_RT_scale_compare.csv";
    file = fopen(fileName, "r"); // open the file in "read" mode
    if (file == NULL) {
        file = fopen(fileName, "w"); //create new file in "write" mode
        fprintf(file, "index_time, rank, date-time, dist file, total_computing_time (s), sending_time (s), BaB_computing_time (s), gathering_time (s), count_BaB, best_bound, best_path, MODE_SEND, MODE_GATHER\n"); // add header to the file
    } else {
        fclose(file);
        file = fopen(fileName, "a"); // open the file in "append" mode
    }

    // Get the current date and time
    strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", &tm);
    fprintf(file, "%f, %d, %s, %s, %f, %f, %f, %f, %f, %f, %f, %d, %d\n", index_time, rank, date, dist_file, total_computing_time, sending_time, BaB_computing_time, gathering_time, count_bab, r_best_bound, r_best_path, MODE_SEND, MODE_GATHER); // add new data to the file

    fclose(file); // close the file
}

double power(double base, int exponent) {
    double result_power = 1;
    while (exponent > 0) {
        if (exponent & 1) result_power *= base;
        base *= base;
        exponent >>= 1;
    }
    return result_power;
}
