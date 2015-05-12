#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <gtk/gtk.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include "defs.h"


int vrender( FLOAT_GRID *, float, float ,float, float, float, float, int );


int LoadImageAsText(FLOAT_GRID *img, const char *path )
{
    FILE *fp;
    char name[255];
    img->max = -10000.0;
    img->min = 32976.0;
    printf(" loading data..... "); fflush(stdout);
    int position = 0;

    for (int k = 1; k < img->count.z; k++)
    {
        sprintf(name, "%s/File%04d.txt", path, k);
        //sprintf(name, "%s/Frame_%04d.txt", path, k);
        printf("\r File %d of %d", k+1, img->count.z); fflush(stdout);
        fp = fopen(name, "r");
        if (fp != NULL)
        {
            for (int j = 0; j < img->count.y; j++)
                for (int i = 0; i < img->count.x; i++)
                {
                    float temp;
                    fscanf(fp,"%f", &temp);
                    if (temp > img->max) img->max = temp;
                    if (temp < img->min) img->min = temp;
                    img->matrix[position] = temp;
                    position = position + 1;
                }
            fclose(fp);
        }
        else
        {
            printf(" LOAD DATA ERROR \n");
            return(FAILURE);
        }
    }

    printf("... loaded. Max/Min value is %f / %f\n", img->max, img->min);
    return(SUCCESS);
}

void copyFloatGrid ( FLOAT_GRID *copy, FLOAT_GRID *data)
{
    copy->count.x = data->count.x;
    copy->count.y = data->count.y;
    copy->count.z = data->count.z;
    copy->inc.x = data->inc.x;
    copy->inc.y = data->inc.y;
    copy->inc.z = data->inc.z;
    copy->max = 1.0f;
    copy->min = 0.0f;
    copy->matrix = (float*)malloc(data->count.x*data->count.y*data->count.z*sizeof(float));
    memset(copy->matrix,0.0f,data->count.x*data->count.y*data->count.z*sizeof(float));
}

void copyParams( FLOAT_GRID *data, int3 dataSize, float3 voxSize )
{
    data->count.x = dataSize.x;
    data->count.y = dataSize.y;
    data->count.z = dataSize.z;
    data->inc.x = voxSize.x;
    data->inc.y = voxSize.y;
    data->inc.z = voxSize.z;
}



/* SORT
    int count2 = count;
    for ( int i=0; i<2*count; i++){
        for( int j=1; j<count2; j++){
            if ( sort[j-1] > sort[j] ){

                int tempint = sort[j];
                sort[j] = sort[j-1];
                sort[j-1] = tempint;

                char tempchar[1024];
                strcpy(tempchar,dfiles->slice[j].name);
                strcpy(dfiles->slice[j].name,dfiles->slice[j-1].name);
                strcpy(dfiles->slice[j-1].name,tempchar);
            }
        }
        count2--;
    }
*/

void display_usage()
{
    printf("\n USAGE:");
    printf("\n   -datafolder='path to directory containing text files'");
    printf("\n   -paramfile='path to file containing data parameters'");
    printf("\n               paramfile format: Dx Dy Dz Vx Vy Vz");
    printf("\n               where D is the data dimensions in voxels");
    printf("\n               and V is the voxel dimensions in mm");
    printf("\n\n OPTIONAL FLAGS:");
    printf("\n   -maskfolder='path to directory containing text files");
    printf("\n                this data will be used to mask the data");
    printf("\n                the mask should have the same dimensions as the data");
    printf("\n                the mask should be binary (0 or 1)");
    printf("\n   -colorscale=1, 2, or 3");
    printf("\n                colorscale 1 is a Gray scale: black->white");
    printf("\n                colorscale 2 is a standard heat map:");
    printf("\n                             black->purple->blue->green->yellow->orange->red->white");
    printf("\n                colorscale 3 is a black centered heat map (for instances with negative values");
    printf("\n                             purple->blue->teal->black->yellow->orange->red");
    printf("\n\n RENDER CONTROLS:");
    printf("\n   Reduce Opacity:            '-' minus");
    printf("\n   Increase Opacity:          '=' equals");
    printf("\n   Reduce Brightness:         '[' left bracket");
    printf("\n   Increase Brightness:       ']' right bracket");
    printf("\n   Shift Offset Negative:     'k' ");
    printf("\n   Shift Offset Positive:     'l' ");
    printf("\n   Narrow Scale:              ',' comma");
    printf("\n   Broaden Scale:             '.' period");
    printf("\n\n");
}


int main(int argc, char *argv[])
{
    // Initialize GTK+
    gtk_init (&argc, &argv);

    FLOAT_GRID *inputMask;
    FLOAT_GRID *inputData;

    FILE *fp;
    int3 dataSize;
    float3 voxelSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        display_usage();
        return 0;
    }


    if (checkCmdLineFlag(argc, (const char **)argv, "datafolder"))
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "paramfile"))
        {
            char *paramFilename;
            getCmdLineArgumentString( argc, (const char**)argv, "paramfile", &paramFilename );
            printf("Parameter File: %s", paramFilename);

            fp = fopen(paramFilename, "r");
            if (fp != NULL)
            {
                fscanf(fp,"%d %d %d %f %f %f",&dataSize.x,&dataSize.y,&dataSize.z,&voxelSize.x,&voxelSize.y,&voxelSize.z);
                fclose(fp);
            }
            else
            {
                printf("Error! No parameter file found,");
            }
        }
        else
        {
            printf("\n Parameter Filed Not Specified. (Usage: -paramfile= )\n\n");
            return 0;
        }
        printf("\n Data Dim: %d x %d x %d",dataSize.x,dataSize.y,dataSize.z);
        printf("\n Voxel Dim: %2.3f x %2.3f x %2.3f\n",voxelSize.x,voxelSize.y,voxelSize.z);
        fflush(stdout);

        char *dataFolder;
        getCmdLineArgumentString( argc, (const char**)argv, "datafolder", &dataFolder );
        printf("\n Data Folder: %s\n", dataFolder);

        inputData = new FLOAT_GRID;
        copyParams( inputData, dataSize, voxelSize );
        inputData->matrix = (float *)malloc(inputData->count.x*inputData->count.y*inputData->count.z*sizeof(float));
        memset(inputData->matrix,0,inputData->count.x*inputData->count.y*inputData->count.z*sizeof(float));

        LoadImageAsText( inputData, dataFolder );

        if (checkCmdLineFlag(argc, (const char **)argv, "maskfolder"))
        {
            char *maskFolder;
            getCmdLineArgumentString( argc, (const char**)argv, "maskfolder", &maskFolder );
            printf("\n Mask Folder: %s\n", maskFolder);

            inputMask = new FLOAT_GRID;
            copyParams( inputMask, dataSize, voxelSize );
            inputMask->matrix = (float *)malloc(inputMask->count.x*inputMask->count.y*inputMask->count.z*sizeof(float));
            memset(inputMask->matrix,0,inputMask->count.x*inputMask->count.y*inputMask->count.z*sizeof(float));

            LoadImageAsText( inputMask, maskFolder );

            for (int i=0; i<inputMask->count.x*inputMask->count.y*inputMask->count.z; i++)
                if (inputMask->matrix[i] == 0)
                    inputData->matrix[i] = 0;

            free(inputMask->matrix);
            delete inputMask;
        }

        int colorScale = 0;
        if (checkCmdLineFlag(argc, (const char **)argv, "colorscale"))
        {
            colorScale = getCmdLineArgumentInt(argc, (const char **)argv, "colorscale");
            if (colorScale <= 1)
                printf("\n Gray Scale Color Map Chosen.\n");
            else if (colorScale == 2)
                printf("\n Standard Heat Map Chosen.\n");
            else if (colorScale == 3)
                printf("\n Center Black Heat Map Chosen.\n");
        }
        vrender( inputData, inputData->max, 0.1f, 1.5f, 0.0f, 1.0f, 1.0f, colorScale );

        free(inputData->matrix);
        delete inputData;
    }
    else
    {
        printf("\n No Data Folder Specified.\n");
        display_usage();
        return 0;
    }

  return(SUCCESS);
}
