#ifndef __H_SAB__
#define __H_SAB__

template<typename depth_t>
struct sab{

    inline static __device__ depth_t get_sab(depth_t fcls, depth_t depth){ // get status array bit
        return ((depth_t) 0xC0000000 & (fcls << 30)) | ((depth_t) 0x3FFFFFFF & (depth));}
    inline static __device__ depth_t get_fcls( depth_t sab){ // get frontier class
        return ((depth_t) 0xC0000000 & sab) >> 30;}
    inline static __device__ depth_t get_depth(depth_t sab){ // get depth
        return ((depth_t) 0x3FFFFFFF & sab);}

    inline static __device__ depth_t clfy_fcls(index_t deg){

        if(deg < th_a)
            return FCLS_TH;

        else if(deg >= th_b)
            return FCLS_MW;

        else
            return FCLS_UW;
    }
};

#endif // __H_SAB__
