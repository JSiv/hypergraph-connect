class TestEval:

    def __init__(self):
        super(TestEval, self).__init__()

    def test(self, opts, generator):
        index = 0
        for items in data_loader:
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            gen_optimizer = optim.Adam(params=generators.parameters(), lr=float(0.0001), betas=(0.0, 0.9))
            gen_optimizer.zero_grad()
            gen_loss = 0

            edges_masked = (edges * (1 - masks))
            images_masked = (images * (1 - masks)) + masks
            inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
            outputs = self.generator(inputs)

            outputs_merged = (outputs * masks) + (edges * (1 - masks))

            outputs_merged_temp = outputs_merged * 255.0
            outputs_merged_temp = outputs_merged_temp.permute(0, 2, 3, 1)

            fromarray(outputs_merged_temp[0].cpu().numpy().astype(np.uint8).squeeze()).imsave(optss.result)

            # if self.debug:
            #     edges = postprocess(1 - edges)[0]
            #     masked = postprocess(images * (1 - masks) + masks)[0]
            #     fname, fext = name.split('.')
            #
            #     imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
            #     imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')