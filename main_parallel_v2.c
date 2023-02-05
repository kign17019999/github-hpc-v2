#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MAX_CITIES 20   // maximum number of cities that can be considered
#define INFINITE INT_MAX    // a large constant used to initiation
#define ROOT 0  // define what is the ROOT processor of the program

#define LOOP_1ST 0 //set 1 to eneble loop all start city from START_CITIES to n
int START_CITIES=9; //start with 0 to n-1

/* MODE_SEND 0 = send Dist by Bcast       | MODE_SEND 1 = send Dist by Ibcast
   MODE_SEND 2 = send Dist by Send & Recv | MODE_SEND 3 = send Dist by Isend & Irecv */
#define MODE_SEND 0

/* MODE_GATHER 0 = gather by Allgather | MODE_GATHER 1 = gather by Send & Recv */
#define MODE_GATHER 0

#define PRINT_ALL 0 //set 1 to eneble print the detial of result (each start_city)
#define SAVE_CSV 1  //set 1 to enable CSV save function
#define NUM_RESULT 7    //number element in array of result

int n;  // number of cities
int (*dist)[MAX_CITIES], (*best_path)[MAX_CITIES];  // 2D array of dist_city and best_path each proc
int *best_path_bound;   //1D array for best path bound each procs

/* city initiation variable */
int (*init_path)[MAX_CITIES], (*init_visited)[MAX_CITIES];  //2D array of init_path and init_visited each procs
int *init_bound, *init_path_rank;   // 1D array of init_path_bound and array of init_rank that tell procs should do what's init_path
int init_position, init_level, init_rank, num_init_path;

//other variable
double (*result)[NUM_RESULT];   // 2D array of result of each procs

#define NUM_BRA_BOU 1 //set 1 to eneble countinf of number of accessing to brach_and_bound function
double count_branch_bound;  // variable to count accessing

/* variable to perform Sharing communication*/
int all_best_bound; // the best_bound of all porc
MPI_Request *global_request_Isend;  //sending request address to track sharing commmunication
MPI_Request *global_request_Irecv;  //gathering request address to track sharing commmunication
int incoming_bound; //variable to store new sharing bound (checking state before save into all_best_bound)
int global_flag;    //flag variable (easu access by all function)

void get_cities_info(char* file_path);
void send_data_to_worker(int rank, int size);
void do_wsp(int rank, int size);
void level_initiation(int size);
void path_initiation(int *path_i, int path_bound, int *visited_i, int level, int size);
void branch_and_bound(int *path, int path_bound, int *visited, int level, int rank, int size);
void gather_result(int rank, int size);
void save_result(int rank, int size, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, char* file_path);
void save_result_csv(double index_time, int rank, char *dist_file, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, double count_bab, double r_best_bound, double r_best_path);
double power(double base, int exponent);

/*  Main function for WSP
    The function performs the following tasks:
    - Sets default file path to "input/distX" or uses the file path provided with "-i" option
    - Calls get_cities_info to parse the distance information of the cities
    - Calls do_wsp which perform level_initiation > path_initiation > branch_and_bound
    - Prints the best path, the best path distance and the computing time
    - Saves the result in a file
*/
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* set the best parameter to print of from loop through
    all starter if needed*/
    int the_best_path_bound=INFINITE;
    int the_best_start, check_best=0;
    double time_of_the_best;
    int *the_best_path = malloc(sizeof(int[MAX_CITIES]));;

    while(1){   // loop all possible start city if needed (if not it will break at the end of first iteration)
        if(rank==ROOT){
            if(PRINT_ALL==1) printf("====================================================\n");
            if(PRINT_ALL==1) if(LOOP_1ST==1) printf("Start with City %d \n", START_CITIES);
        }

        double start_time1 = MPI_Wtime();   //total time

        MPI_Request request1, request2;

        /* set default array */
        dist = malloc(sizeof(int[MAX_CITIES][MAX_CITIES])); //city information
        best_path = malloc(sizeof(int[size][MAX_CITIES]));  //size of array depend on number of procs
        best_path_bound = malloc(sizeof(int[size]));    //size of best_bound depend on number of procs
        best_path_bound[rank] = INFINITE;   //initiate by infinity

        /* retrive city information */
        char* file_path;
        if(argc >=3 && strcmp("-i", argv[1]) == 0){ // check there are enough input to consider or not
            char* myArg = argv[2];
            while (myArg[0] == '\'') myArg++;   // filter some input mistake at the front of file name
            while (myArg[strlen(myArg)-1] == '\'') myArg[strlen(myArg)-1] = '\0';   // filter some input mistake at the last of file name
            file_path = myArg;
        }else{  // if there are no input of filename
            char *df_file = "input/dist10";
            if(PRINT_ALL==1) if (rank==ROOT) printf("    [ROOT] The default file (%s) will be used if no input is provided  \n", df_file);
            file_path  = df_file;
        }
        

        if(rank==ROOT){ // compute saving data from file to program
            get_cities_info(file_path);
            if(PRINT_ALL==1) printf("    [ROOT] number of cities = %d \n", n);
            if(PRINT_ALL==1) printf("    [ROOT] number of processor = %d \n", size);
        }

        //send city data to all processor
        double start_time2 = MPI_Wtime();   // timing for sending communication
        send_data_to_worker(rank, size);
        double end_time2 = MPI_Wtime();

        //solving wsp
        double start_time3 = MPI_Wtime();   // timing for wsp task
        do_wsp(rank, size);
        double end_time3 = MPI_Wtime();

        //gathering result
        double start_time4 = MPI_Wtime();   // timing for gathering communication
        gather_result(rank, size);
        double end_time4 = MPI_Wtime();

        /* update the best result from all procs*/
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

            /* update the best of all starting city */
            if(the_best_path_bound>best_path_bound[index_best_path]){
                the_best_path_bound = best_path_bound[index_best_path];
                the_best_start = START_CITIES;
                for(int i=0; i<n; i++) the_best_path[i]=best_path[index_best_path][i];
                check_best=1;
            }

        }

        double end_time1 = MPI_Wtime(); //total time

        /* compute timing & printing the result */
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
        if(check_best==1){  // update best time of starting city 
            time_of_the_best = total_computing_time;
            check_best=0;
        }

        /* save result to the file */
        if(SAVE_CSV==1) save_result(rank, size, total_computing_time, sending_time, BaB_computing_time, gathering_time, file_path);
     
        /* free memory */
        free(dist);
        free(best_path);
        free(best_path_bound);
        free(init_path);
        free(init_bound);
        free(init_visited);
        free(init_path_rank);
        
        /* check user need to loop or starting city or not */
        if(LOOP_1ST==1){
            START_CITIES++;
            if(START_CITIES==n) break;  // if there are no further starting city > let break
        }else break;    //if not break here
    }

    /* print the best result of all starting city */
    if(rank==ROOT){
        printf("best path bound = %d | total time = %f s | path = ", the_best_path_bound, time_of_the_best);
        for(int i=0; i<n; i++) printf("%d ", the_best_path[i]);
        printf("| RT = TRUE");
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}

/*  get_cities_info: reads information about the cities and their distances 
    from a file and stores it in the 2D array "dist".
    Input: file_path - a string specifying the file path of the input file.
*/
void get_cities_info(char* file_path) {
    FILE* file = fopen(file_path, "r");
    char line[256];
    fscanf(file, "%d", &n);     //read number of city by the fist element in file

    int row=1;
    int col=0;
    dist[row-1][col]=0; //intial first value of dist array is 0 because it is itself travelled

    while (fgets(line, sizeof(line), file)) {
        col=0;
        char *token = strtok(line, " ");    //get element information by checking " " as a seperater
        while (token != NULL) {
            if(atoi(token)>0){
                dist[row-1][col] = atoi(token);
                dist[col][row-1] = atoi(token);
            }
            token = strtok(NULL, " ");
            col++;  // go to next elemement in a row
        }
        dist[row][col]=0;
        row++;  //go to next row
    }
    fclose(file);
}

/*  sending communication depending specify communicatin mode
*/
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

/* perform level_initiation > path_initiation > branch_and_bound
*/
void do_wsp(int rank, int size){
    //set initial value
    init_position = init_level = init_rank = count_branch_bound = 0;
    num_init_path=1;
    if(START_CITIES>=n) START_CITIES=0; //if START_CITIES is larger than n > make start with city 0

    //path and visit for initiation
    int *path_i = malloc(MAX_CITIES * sizeof(int));
    int *visited_i = malloc(MAX_CITIES * sizeof(int));
    for(int i=0; i<MAX_CITIES; i++) visited_i[i]=0; //intiate vistied with value 0 (unvisited)
    path_i[0] = START_CITIES;   //assign first city to START_CITIES
    visited_i[START_CITIES] = 1;    //mark that START_CITIES has been visited
    level_initiation(size); //calcuate initiate level (which level should enough to distribute task to all processor)

    //path initiation
    init_path=malloc(sizeof(int[num_init_path][MAX_CITIES]));
    init_bound=malloc(num_init_path * sizeof(int));
    init_visited=malloc(sizeof(int[num_init_path][MAX_CITIES]));
    init_path_rank=malloc(num_init_path * sizeof(int));
    path_initiation(path_i, 0, visited_i, 1, size);    //decompose task into array of initiation path    

    /* intiate Sharing communication by read and send some data to start the sharing commu w/o check existing of request */
    all_best_bound = INFINITE;  // start with infinite bound
    global_request_Isend = malloc(size * sizeof(MPI_Request));  //define array sie of sending request
    global_request_Irecv = malloc(size * sizeof(MPI_Request));  //define array size of sending request
    for(int i=0; i<size; i++){  // looping to send all and get or data
        if(i!=rank){
            MPI_Isend(&all_best_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Isend[i]);
            MPI_Irecv(&incoming_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Irecv[i]);
        }
    }

    //path and visit for branch_and_bound
    int *path = malloc(MAX_CITIES * sizeof(int));
    int *visited = malloc(MAX_CITIES * sizeof(int));
    for(int i=0; i<num_init_path; i++){
        if(init_path_rank[i]==rank){    //check which initate that the procs should perform refer to init_rank
            for(int j=0; j<n; j++){     //copy init_path into brach and bound path
                path[j] = init_path[i][j];
                visited[j] = init_visited[i][j];
            }
            branch_and_bound(path, init_bound[i], visited, init_level, rank, size);     // compute branch_and_bound    
        }
        
    }

    /* free memory */
    free(path_i);
    free(visited_i);
    free(path);
    free(visited);
    free(global_request_Irecv);
    free(global_request_Isend);

}

/*  calcuate a suitable level that the system can distribute task to all procs equally
*/
void level_initiation(int size){
    for(int level=1; level<n; level++){
        num_init_path = num_init_path*(n-level);    // number of initiation path in this level
        if(num_init_path >= size || level==n-1){    // if number of initiation path is larger than  number of procs > enough!
            if(level==n-1) init_level=level;
            else init_level=level+1;
            break;
        }
    }
}

/*  do branch and bound to save intiate path until reaching a proper init_level
*/
void path_initiation(int *path_i, int path_bound, int *visited_i, int level, int size) {
    if (level == init_level) {  // check that the travelling is going to reach the end of init_level or not
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
            if (!visited_i[i]) {    // check there are unvisited city persist or not
                path_i[level] = i;  // mark what is the path we are going to evaluate
                visited_i[i] = 1;   // mark the currect city is visited
                int new_bound = path_bound + dist[i][path_i[level - 1]];    // calculate new bound
                path_initiation(path_i, new_bound, visited_i, level + 1, size);
                visited_i[i] = 0;   // reset the visited index, prepare to check new unvisited city
            }
        }
    }
}

/*  branch_and_bound is a recursive function to calculate the best path bound for the WSP
    path: array to store the order of visiting cities
    path_bound: current bound of the path
    visited: array to store the visited status of the cities
    level: current level (city) of visiting
    rank: rank of proc in the system
    sizeL number of proc in the system
*/
void branch_and_bound(int *path, int path_bound, int *visited, int level, int rank, int size) {
    if(NUM_BRA_BOU==1) count_branch_bound+=1;   // incease number of access to branch and bound
    if (level == n) {   // check that the travelling is going to the end of path or not
        for(int i=0; i<size;i++){   // check the current global new best path bound from all procs
            if(i!=rank){
                global_flag=0;
                MPI_Test(&global_request_Irecv[i], &global_flag, MPI_STATUS_IGNORE);    //check the status of previos msg recv
                if(global_flag){
                    if(incoming_bound < all_best_bound){
                      MPI_Cancel(&global_request_Isend[i]);    // if the new global bound is better than prebious information it has > cancle previous sharing
                      all_best_bound = incoming_bound;
                    }
                    MPI_Irecv(&incoming_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Irecv[i]); // read new update (inadvance! > it will no data until other procs send something > it will save into incomming)
                }
            }
        }
        if (path_bound < all_best_bound) {  //check the new path is better than the previous best or not
            best_path_bound[rank] = path_bound;
            all_best_bound = path_bound;
            for(int i = 0; i < n; i++) best_path[rank][i] = path[i]+1;
            for(int i = 0; i < size; i++) { // if new calucate found new best >> sharing the best to all procs
                if(i != rank) {
                    MPI_Cancel(&global_request_Isend[i]);   // cancel previous send (if existing) (MPI_cancel will not block any process)
                    MPI_Isend(&all_best_bound, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &global_request_Isend[i]); // sharing
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {  // check there are unvisited city persist or not
                path[level] = i;    // mark what is the path we are going to evaluate
                visited[i] = 1; // mark the currect city is visited
                int new_bound = path_bound + dist[i][path[level - 1]];  // calculate new bound
            if (new_bound < all_best_bound) branch_and_bound(path, new_bound, visited, level + 1, rank, size);
            visited[i] = 0; // reset the visited index, prepare to check new unvisited city
	        }
        }
    }
}

/*  Gathering communication depending specify communicatin mode
*/
void gather_result(int rank, int size){
    int send_buf_bound = best_path_bound[rank]; // set variable to store send buf (avoid buf issue)
    int row_to_gather[MAX_CITIES];  // set variable to store send buf (avoid buf issue)
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

/*  function share result among all procs */
void save_result(int rank, int size, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, char* file_path){
    result = malloc(sizeof(double[size][NUM_RESULT]));  // 2D array of result depend on number of procs
    result[rank][0] = total_computing_time;
    result[rank][1] = sending_time;
    result[rank][2] = BaB_computing_time;
    result[rank][3] = gathering_time;
    result[rank][4] = count_branch_bound;
    result[rank][5] = best_path_bound[rank];
    double double_path = power(10, 2*n)*404;    /* make best path into a double format that can save into the same result array which is a double type*/
    for(int i=0; i<n; i++){ // add each city order into a large number sequently
        double_path+=power(10, (n-i-1)*2)*best_path[rank][i];
    }
    result[rank][6] = double_path;
    
    // set buf of sending result (avoid duff issue)
    double row_to_gather_result[NUM_RESULT];    
    for (int i = 0; i < NUM_RESULT; i++) row_to_gather_result[i] = result[rank][i];

    // allgather result to all procs
    MPI_Allgather(row_to_gather_result, NUM_RESULT, MPI_DOUBLE  , result, NUM_RESULT, MPI_DOUBLE  , MPI_COMM_WORLD);

    if(rank==ROOT){ // only save the result if rank = ROOT (avoid file access problem)
        double index_time = MPI_Wtime();
        for(int i=0; i<size;i++) save_result_csv(index_time, i, file_path, result[i][0], result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]);
    }

    /* free memory */
    free(result);
}

/*  This function saves WSP results to a .csv file. */
void save_result_csv(double index_time, int rank, char *dist_file, double total_computing_time, double sending_time, double BaB_computing_time, double gathering_time, double count_bab, double r_best_bound, double r_best_path) {
    FILE *file;
    char date[20];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    char* fileName="result_parallel_RT.csv";
    file = fopen(fileName, "r"); // open the file in "read" mode
    if (file == NULL) { // check there are existing file or not
        file = fopen(fileName, "w"); //create new file in "write" mode
        fprintf(file, "index_time, rank, date-time, dist file, total_computing_time (s), sending_time (s), BaB_computing_time (s), gathering_time (s), count_BaB, best_bound, best_path, MODE_SEND, MODE_GATHER\n"); // add header to the file
    } else {    // if no any existing file >> create the new file
        fclose(file);
        file = fopen(fileName, "a"); // open the file in "append" mode
    }

    // Get the current date and time
    strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", &tm);
    fprintf(file, "%f, %d, %s, %s, %f, %f, %f, %f, %f, %f, %f, %d, %d\n", index_time, rank, date, dist_file, total_computing_time, sending_time, BaB_computing_time, gathering_time, count_bab, r_best_bound, r_best_path, MODE_SEND, MODE_GATHER); // add new data to the file

    fclose(file); // close the file
}

/* This function calculates the power of a base number to an exponent.*/
double power(double base, int exponent) {
    double result_power = 1;
    while (exponent > 0) {
        if (exponent & 1) result_power *= base;
        base *= base;
        exponent >>= 1;
    }
    return result_power;
}
