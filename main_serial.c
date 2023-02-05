#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <string.h>

#define MAX_CITIES 20   // maximum number of cities that can be considered
#define INFINITE INT_MAX // a large constant used to initiation
#define START_CITIES 9 //start with 0 to n-1

int n; // number of cities
int (*dist)[MAX_CITIES]; // a 2D array to store distances between cities
int *best_path; // an array to store the best path found so far
int best_path_bound = INFINITE; // the length of the best path found so far

void get_cities_info(char* file_path);
void branch_and_bound(int *path, int path_bound, int *visited, int level);
void save_result(char* dist_file, double computing_time);
double power(double base, int exponent);

/*  Main function for WSP
    The function performs the following tasks:
    - Sets default file path to "input/distX" or uses the file path provided with "-i" option
    - Calls get_cities_info to parse the distance information of the cities
    - Calls branch_and_bound to find the shortest path that visits all cities
    - Prints the best path, the best path distance and the computing time
    - Saves the result in a file named after the input file and computing time 
*/
int main(int argc, char *argv[]) {
    clock_t start = clock();

    /* set default array to the size of MAX_CITIES */
    dist = malloc(sizeof(int[MAX_CITIES][MAX_CITIES]));
    best_path = malloc(sizeof(int[MAX_CITIES]));

    /* retrive city information */
    char* file_path;
    if(argc >=3 && strcmp("-i", argv[1]) == 0){
        char* myArg = argv[2];
        while (myArg[0] == '\'') myArg++;
        while (myArg[strlen(myArg)-1] == '\'') myArg[strlen(myArg)-1] = '\0';;
        file_path = myArg;
    }else{
        char *df_file = "input/dist13";
        printf("[System] The default file (%s) will be used if no input is provided  \n", df_file);
        file_path  = df_file;
    }
    get_cities_info(file_path);
    
    /* initial path and vistied array */
    int *path = malloc(MAX_CITIES * sizeof(int));
    int *visited = malloc(MAX_CITIES * sizeof(int));
    for(int i=0; i<MAX_CITIES; i++) visited[i]=0;   //intiate vistied with value 0 (unvisited)
    path[0] = START_CITIES; //assign first city to START_CITIES
    visited[START_CITIES] = 1;  //mark that START_CITIES has been visited

    /* compute branch_and_bound */
    branch_and_bound(path, 0, visited, 1);
    double computing_time = (double)(clock() - start) / CLOCKS_PER_SEC;

    /* print the result */
    printf("best path bound = %d | total time = %f s | path = ", best_path_bound, computing_time);
    for(int i=0; i<n; i++) printf("%d ", best_path[i]);
    printf("| Mode = Serial");
    printf("\n");
    
    /* save result to the file */
    save_result(file_path, computing_time);

    /* free memory */
    free(dist);
    free(best_path);
    return 0;
}
/*  get_cities_info: reads information about the cities and their distances 
    from a file and stores it in the 2D array "dist".
    Input: file_path - a string specifying the file path of the input file.
*/
void get_cities_info(char* file_path) {
    FILE* file = fopen(file_path, "r");
    char line[256];
    fscanf(file, "%d", &n); //read number of city by the fist element in file

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

/*  branch_and_bound is a recursive function to calculate the best path bound for the WSP
    path: array to store the order of visiting cities
    path_bound: current bound of the path
    visited: array to store the visited status of the cities
    level: current level (city) of visiting */
void branch_and_bound(int *path, int path_bound, int *visited, int level) {
    if (level == n) {   // check that the travelling is going to the end of path or not
        if (path_bound < best_path_bound) {
            best_path_bound = path_bound;
            for (int i = 0; i < n; i++) best_path[i] = path[i]+1;
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {  // check there are unvisited city persist or not
                path[level] = i;
                visited[i] = 1;
                int new_bound = path_bound + dist[i][path[level - 1]];
                if (new_bound < best_path_bound) branch_and_bound(path, new_bound, visited, level + 1);
                visited[i] = 0;
            }
        }
    }
}

void save_result(char* dist_file, double computing_time) {
    double double_path = power(10, 2*n)*404;
    for(int i=0; i<n; i++){
        double_path+=power(10, (n-i-1)*2)*best_path[i];
    }

    FILE *file;
    char date[20];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    char* fileName="result_serial.csv";
    file = fopen(fileName, "r"); // open the file in "read" mode
    if (file == NULL) {
        file = fopen(fileName, "w"); //create new file in "write" mode
        fprintf(file, "date-time, dist file, computing time (s), best_bound, best_path\n"); // add header to the file
    } else {
        fclose(file);
        file = fopen(fileName, "a"); // open the file in "append" mode
    }

    // Get the current date and time
    strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", &tm);
    fprintf(file, "%s,%s,%f, %d, %f\n", date, dist_file, computing_time, best_path_bound, double_path); // add new data to the file

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