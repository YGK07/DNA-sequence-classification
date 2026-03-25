#include <stdio.h>
#include <string.h>

void naiveMatch(char dna[], char pattern[]) {
    int n = strlen(dna);
    int m = strlen(pattern);
    int found = 0;
    for(int i = 0; i <= n - m; i++) {
        int j;
        for(j = 0; j < m; j++) {
            if(dna[i + j] != pattern[j])
                break;
        }
        if(j == m) {
            printf("Pattern found at position %d\n", i);
            found = 1;
        }
    }
    if(!found)
        printf("Pattern not found in DNA sequence\n");
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        printf("Usage: %s <dna_sequence> <pattern>\n", argv[0]);
        return 1;
    }
    char *dna = argv[1];
    char *pattern = argv[2];
    printf("\n=== Naive String Matching ===\n");
    printf("DNA Sequence : %s\n", dna);
    printf("Gene Pattern : %s\n\n", pattern);
    naiveMatch(dna, pattern);
    return 0;
}
