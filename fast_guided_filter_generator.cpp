#include <Halide.h>

using namespace Halide;

class FastGuidedFilter : public Halide::Generator<FastGuidedFilter>{
    public:

        GeneratorParam<int> radius{"radius",8,1,100};
        Input<Buffer<uint8_t>> guided{"guided", 3},input{"input", 3};
        Input<float> eps{"eps"};
        //@TODO set radius as Input param is really slow.
        //Input<int> radius{"radius"};
        //@TODO still dont know how to write a [up/down]sample func with,arbitrary scale factor.
        //Input<int> s{"s"};
        Output<Buffer<uint8_t>> result{"result", 3};
 
        void generate() {
            Var x("x"),y("y"),c("c");

            Func input_c = BoundaryConditions::mirror_image(input);
            Func guided_c = BoundaryConditions::mirror_image(guided);

            Expr rad = cast<int>(radius/2);
            Expr size = 2*rad+1;
            Expr area = size*size;
            Expr eps_ = eps*area;
            Expr area_ = 1.0f/area;
            Expr fac = 255.0f/area;


            Halide::RDom rb(-rad, size, "rb");

            Halide::Func inputF("inputFloat"),guidedF("guidedFloat");
            Halide::Func mean_I("mean_I"),mean_p("mean_p"),
                         II("II"),Ip("IP"),corr_I("corr_I"),corr_Ip("corr_Ip"),
                         cov_Ip("cov_Ip"),var_I("var_I"),
                         a("a"),b("b"),mean_a("mean_a"),mean_b("mean_b"),q("q");

            Halide::Func mean_I_sx("mean_I_sx"),corr_I_sx("corr_I_sx"),mean_p_sx("mean_p_sx"),
                         corr_Ip_sx("corr_Ip_sx"),mean_a_sx("mean_a_sx"),mean_b_sx("mean_b_sx");

            Func inputSub("inputSub"),guidedSub("guidedSub"),
                mean_aUp("mean_aUp"),mean_bUp("mean_bUp");

            inputF(x,y,c) = Halide::cast<float>(input_c(x,y,c)/255.0f);
            guidedF(x,y,c) = Halide::cast<float>(guided_c(x,y,c));

            inputSub(x,y,c) = downsample(inputF)(x,y,c);
            guidedSub(x,y,c) = downsample(guidedF)(x,y,c);

            mean_I_sx(x,y,c) = sum(guidedSub(x+rb, y, c));
            mean_I(x,y,c) = sum(mean_I_sx(x,y+rb,c));


            II(x,y,c) = guidedSub(x,y,c)*guidedSub(x,y,c);
            Ip(x,y,c) = inputSub(x,y,c)*guidedSub(x,y,c);

            corr_I_sx(x,y,c) = sum(II(x+rb,y,c));
            corr_I(x,y,c) = sum(corr_I_sx(x,y+rb,c));

            mean_p_sx(x,y,c) = sum(inputSub(x+rb,y,c));
            mean_p(x,y,c) = sum(mean_p_sx(x,y+rb,c));

            corr_Ip_sx(x,y,c) = sum(Ip(x+rb,y,c));
            corr_Ip(x,y,c) = sum(corr_Ip_sx(x,y+rb,c));

            cov_Ip(x,y,c) = corr_Ip(x,y,c)-mean_I(x,y,c)*mean_p(x,y,c)*area_;

            var_I(x,y,c) = corr_I(x,y,c)-mean_I(x,y,c)*mean_I(x,y,c)*area_;

            a(x,y,c) = cov_Ip(x,y,c)/(var_I(x,y,c)+eps_);

            b(x,y,c) = (mean_p(x,y,c)-a(x,y,c)*mean_I(x,y,c))*area_;

            mean_a_sx(x,y,c) = sum(a(x+rb,y,c)); 
            mean_a(x,y,c) = sum(mean_a_sx(x,y+rb,c));

            mean_b_sx(x,y,c) = sum(b(x+rb,y,c)); 
            mean_b(x,y,c) = sum(mean_b_sx(x,y+rb,c));

            mean_aUp(x,y,c) = upsample(mean_a)(x,y,c);
            mean_bUp(x,y,c) = upsample(mean_b)(x,y,c);


            q(x,y,c) = mean_aUp(x,y,c)*guidedF(x,y,c) + mean_bUp(x,y,c);
            result(x,y,c) = Halide::cast<uint8_t>(Halide::clamp(q(x,y,c)*fac, 0.0f, 255.0f));

            // Provide estimates on the inputs
            guided.set_estimates({{0, 512}, {0, 512}, {0, 3}});
            input.set_estimates({{0, 512}, {0, 512}, {0, 3}});
            // Provide estimates on the parameters
            eps.set_estimate(0.0004);
            // Provide estimates on the pipeline output
            result.set_estimates({{0, 512}, {0, 512}, {0, 3}});

            /*/================
            Image Size:512x512x3
            CPU:I5-3210m
            --------------------
            Opencv: 32ms
            Halide manually-tuned: 41ms
            Halide auto-schedule: 6ms
            ================/*/ 

            if(auto_schedule){
                //nothing
            }
            else{
                int vec=8;
                Var x_outer, y_outer, x_inner, y_inner, tile_index;
                result
                    .tile(x,y,x_inner,y_inner, 64,64)
                    .parallel(y)
                    ;

                mean_b.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                mean_b_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;
                
                mean_a.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                mean_a_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;
                
                a.compute_root();
                b.compute_at(mean_b_sx, x)
                .vectorize(x,vec);
                
                corr_Ip.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                corr_Ip_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;
                Ip.compute_at(corr_Ip_sx,x)
                .vectorize(x,vec);
                
                corr_I.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                corr_I_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;
                II.compute_at(corr_I_sx,x)
                .vectorize(x,vec);
                
                mean_p.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                mean_p_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;
                
                mean_I.compute_root()
                    .tile(x,y,x_inner,y_inner,64,64)
                    .parallel(y);
                mean_I_sx.compute_root()
                    .tile(x,y,x_inner,y_inner,128,64)
                    .reorder(y_inner,x_inner)
                    .parallel(y)
                    ;

                inputF.compute_root();
                guidedF.compute_root();
            }

        }
    private:
        //@TODO Copy from local_laplacian_generator.cpp
        Var x, y, c, k;
        // Downsample with a 1 3 3 1 filter
        Func downsample(Func f) {
            using Halide::_;
            Func downx, downy;
            downx(x, y, _) = (f(2 * x - 1, y, _) + 3.0f * (f(2 * x, y, _) + f(2 * x + 1, y, _)) + f(2 * x + 2, y, _)) / 8.0f;
            downy(x, y, _) = (downx(x, 2 * y - 1, _) + 3.0f * (downx(x, 2 * y, _) + downx(x, 2 * y + 1, _)) + downx(x, 2 * y + 2, _)) / 8.0f;
            return downy;
        }

        // Upsample using bilinear interpolation
        Func upsample(Func f) {
            using Halide::_;
            Func upx, upy;
            upx(x, y, _) = lerp(f(x / 2, y, _), f((x + 1) / 2, y, _), ((x % 2) * 2 + 1) / 4.0f);
            upy(x, y, _) = lerp(upx(x, y / 2, _), upx(x, (y + 1) / 2, _), ((y % 2) * 2 + 1) / 4.0f);
            return upy;
        }
};

HALIDE_REGISTER_GENERATOR(FastGuidedFilter, fast_guided_filter)