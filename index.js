const fs = require("fs");
require("dotenv").config();
const octokit = require("@octokit/core");

const client = new octokit.Octokit({ auth: process.env.GH_TOKEN });

async function updateReadMe(repo) {
	try {
		const res = await client.request(`GET /dcolinmorgan/mlx_grph/contents/README.md`);
		const { path, sha, content, encoding } = res.data;
		const rawContent = Buffer.from(content, encoding).toString();
		const startIndex = rawContent.indexOf("## Other Projects");
		const updatedContent = `${startIndex === -1 ? rawContent : rawContent.slice(0, startIndex)}\n${getNewProjectSection()}`;
		commitNewReadme(repo, path, sha, encoding, updatedContent);
	} catch (error) {
		try {
			const content = `\n${getNewProjectSection()}`;
			await client.request(`PUT /dcolinmorgan/mlx_grph/contents/README.md`, {
				message: "Create README",
				content: Buffer.from(content, "utf-8").toString(encoding),
			});
		} catch (err) {
			console.log(err);
		}
	}
}

async function commitNewReadme(repo, path, sha, encoding, updatedContent) {
	try {
		await client.request(`PUT /dcolinmorgan/mlx_grph/contents/{path}`, {
			message: "Update README",
			content: Buffer.from(updatedContent, "utf-8").toString(encoding),
			path,
			sha,
		});
	} catch (err) {
		console.log(err);
	}
}


updateAllRepos();
