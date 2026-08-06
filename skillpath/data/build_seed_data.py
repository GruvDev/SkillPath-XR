"""
Builds the bundled seed dataset for SkillPath-XR.

WHY THIS EXISTS
---------------
The real ESCO taxonomy (~13,900 skills) is distributed by the European
Commission under its own terms and is too large to bundle.  This script
produces a small, ESCO-*shaped* taxonomy so the system is runnable and
testable with no downloads, and so unit tests have deterministic fixtures.

The shape is faithful to ESCO: every concept has a URI, a preferred label,
alternative labels, a `broader` hierarchy and `related` lateral links, and
occupations declare `essential` and `optional` skills.  Because the shape
matches, swapping in the real taxonomy is a loader change, not a code change
(see skillpath/ontology/esco.py :: load_esco_csv).

Run:  python -m skillpath.data.build_seed_data
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

SKILL_NS = "urn:skillpath:skill:"
OCC_NS = "urn:skillpath:occupation:"


def uri(slug: str) -> str:
    return SKILL_NS + slug


def occ_uri(slug: str) -> str:
    return OCC_NS + slug


# ---------------------------------------------------------------------------
# Skill taxonomy
# (slug, label, type, alt_labels, broader_slugs, related_slugs)
# ---------------------------------------------------------------------------

SKILLS: List[Tuple[str, str, str, List[str], List[str], List[str]]] = [
    # --- top-level umbrella concepts ---------------------------------------
    ("programming", "computer programming", "knowledge",
     ["coding", "software programming", "programming"], [], []),
    ("web-development", "web development", "knowledge",
     ["web dev"], [], []),
    ("data-engineering", "data engineering", "knowledge",
     ["data pipelines"], [], []),
    ("machine-learning", "machine learning", "knowledge",
     ["ML", "statistical learning"], [], []),
    ("cloud-computing", "cloud computing", "knowledge",
     ["cloud"], [], []),
    ("databases", "database management", "knowledge",
     ["databases", "database systems"], [], []),
    ("devops", "DevOps practices", "knowledge",
     ["devops"], [], []),
    ("cybersecurity", "cybersecurity", "knowledge",
     ["information security", "infosec"], [], []),
    ("mobile-development", "mobile application development", "knowledge",
     ["mobile development"], [], []),
    ("product-design", "product design", "knowledge",
     ["digital product design"], [], []),
    ("software-testing", "software testing", "knowledge",
     ["quality assurance", "QA"], [], []),

    # --- programming languages ---------------------------------------------
    ("python", "Python", "knowledge",
     ["python3", "python 3"], ["programming"], ["pandas", "numpy", "django", "flask"]),
    ("java", "Java", "knowledge",
     ["java se", "core java"], ["programming"], ["spring-boot"]),
    ("javascript", "JavaScript", "knowledge",
     ["js", "ecmascript"], ["programming"], ["typescript", "react", "nodejs"]),
    ("typescript", "TypeScript", "knowledge",
     ["ts"], ["programming"], ["javascript", "react", "angular"]),
    ("cpp", "C++", "knowledge",
     ["cplusplus", "c plus plus"], ["programming"], []),
    ("golang", "Go", "knowledge",
     ["golang"], ["programming"], ["microservices"]),
    ("rust", "Rust", "knowledge",
     [], ["programming"], ["cpp"]),
    ("sql", "SQL", "knowledge",
     ["structured query language"], ["programming", "databases"],
     ["postgresql", "mysql", "data-analysis"]),
    ("r-lang", "R", "knowledge",
     ["r programming"], ["programming"], ["statistics", "data-analysis"]),
    ("scala", "Scala", "knowledge",
     [], ["programming"], ["spark"]),

    # --- frontend -----------------------------------------------------------
    ("html", "HTML", "knowledge",
     ["html5"], ["web-development"], ["css"]),
    ("css", "CSS", "knowledge",
     ["css3", "cascading style sheets"], ["web-development"], ["html", "tailwind"]),
    ("react", "React", "knowledge",
     ["reactjs", "react.js"], ["web-development"], ["javascript", "typescript", "nextjs"]),
    ("vue", "Vue.js", "knowledge",
     ["vuejs", "vue"], ["web-development"], ["javascript"]),
    ("angular", "Angular", "knowledge",
     ["angularjs"], ["web-development"], ["typescript"]),
    ("nextjs", "Next.js", "knowledge",
     ["nextjs", "next js"], ["web-development"], ["react"]),
    ("tailwind", "Tailwind CSS", "knowledge",
     ["tailwindcss"], ["web-development"], ["css"]),
    ("web-accessibility", "web accessibility", "skill",
     ["a11y", "accessibility", "wcag"], ["web-development"], ["css", "ux-research"]),

    # --- backend ------------------------------------------------------------
    ("nodejs", "Node.js", "knowledge",
     ["node", "nodejs"], ["web-development"], ["javascript", "rest-api"]),
    ("django", "Django", "knowledge",
     [], ["web-development"], ["python", "rest-api"]),
    ("flask", "Flask", "knowledge",
     [], ["web-development"], ["python", "rest-api"]),
    ("fastapi", "FastAPI", "knowledge",
     [], ["web-development"], ["python", "rest-api"]),
    ("spring-boot", "Spring Boot", "knowledge",
     ["springboot", "spring"], ["web-development"], ["java", "rest-api"]),
    ("rest-api", "REST API design", "skill",
     ["rest", "restful api", "api design"], ["web-development"], ["graphql", "microservices"]),
    ("graphql", "GraphQL", "knowledge",
     [], ["web-development"], ["rest-api"]),
    ("microservices", "microservices architecture", "knowledge",
     ["microservice architecture", "microservices"], ["web-development"],
     ["docker", "kubernetes", "rest-api"]),

    # --- data ---------------------------------------------------------------
    ("data-analysis", "data analysis", "skill",
     ["analysing data", "data analytics"], ["data-engineering"],
     ["sql", "statistics", "data-visualisation"]),
    ("data-visualisation", "data visualisation", "skill",
     ["data visualization", "dataviz", "charting"], ["data-engineering"],
     ["data-analysis", "data-storytelling"]),
    ("pandas", "pandas", "knowledge",
     ["pandas library"], ["data-engineering"], ["python", "numpy", "data-analysis"]),
    ("numpy", "NumPy", "knowledge",
     ["numpy library"], ["data-engineering"], ["python", "pandas"]),
    ("etl", "ETL pipeline development", "skill",
     ["etl", "extract transform load", "elt"], ["data-engineering"],
     ["airflow", "data-warehousing"]),
    ("airflow", "Apache Airflow", "knowledge",
     ["airflow"], ["data-engineering"], ["etl", "python"]),
    ("spark", "Apache Spark", "knowledge",
     ["spark", "pyspark"], ["data-engineering"], ["scala", "hadoop", "etl"]),
    ("hadoop", "Apache Hadoop", "knowledge",
     ["hadoop", "hdfs"], ["data-engineering"], ["spark"]),
    ("kafka", "Apache Kafka", "knowledge",
     ["kafka"], ["data-engineering"], ["microservices", "etl"]),
    ("data-warehousing", "data warehousing", "knowledge",
     ["data warehouse", "dwh"], ["data-engineering"], ["sql", "etl"]),
    ("dbt", "dbt", "knowledge",
     ["data build tool"], ["data-engineering"], ["sql", "data-warehousing"]),
    ("statistics", "statistics", "knowledge",
     ["statistical analysis", "applied statistics"], [],
     ["data-analysis", "machine-learning"]),

    # --- machine learning ---------------------------------------------------
    ("deep-learning", "deep learning", "knowledge",
     ["neural networks", "deep neural networks"], ["machine-learning"],
     ["pytorch", "tensorflow"]),
    ("nlp", "natural language processing", "knowledge",
     ["nlp", "text mining", "computational linguistics"], ["machine-learning"],
     ["llm", "deep-learning"]),
    ("computer-vision", "computer vision", "knowledge",
     ["cv", "image processing"], ["machine-learning"], ["deep-learning"]),
    ("pytorch", "PyTorch", "knowledge",
     ["torch"], ["machine-learning"], ["deep-learning", "python"]),
    ("tensorflow", "TensorFlow", "knowledge",
     ["tf", "keras"], ["machine-learning"], ["deep-learning", "python"]),
    ("scikit-learn", "scikit-learn", "knowledge",
     ["sklearn", "scikit learn"], ["machine-learning"], ["python", "machine-learning"]),
    ("llm", "large language models", "knowledge",
     ["llms", "large language model", "foundation models"], ["machine-learning"],
     ["nlp", "rag", "prompt-engineering"]),
    ("rag", "retrieval-augmented generation", "knowledge",
     ["rag", "retrieval augmented generation"], ["machine-learning"],
     ["llm", "vector-search", "knowledge-graphs"]),
    ("prompt-engineering", "prompt engineering", "skill",
     ["prompting"], ["machine-learning"], ["llm"]),
    ("knowledge-graphs", "knowledge graphs", "knowledge",
     ["knowledge graph", "kg", "semantic graph"], [],
     ["ontology-engineering", "neo4j", "rag"]),
    ("ontology-engineering", "ontology engineering", "skill",
     ["ontology design", "taxonomy design"], [], ["knowledge-graphs"]),
    ("vector-search", "vector similarity search", "skill",
     ["vector search", "embedding search", "semantic search"], [],
     ["rag", "elasticsearch"]),
    ("mlops", "MLOps", "skill",
     ["ml ops", "machine learning operations"], ["machine-learning"],
     ["docker", "ci-cd", "model-evaluation"]),
    ("feature-engineering", "feature engineering", "skill",
     ["feature extraction"], ["machine-learning"], ["data-analysis"]),
    ("model-evaluation", "model evaluation", "skill",
     ["model validation", "model assessment"], ["machine-learning"], ["statistics"]),
    ("recommender-systems", "recommender systems", "knowledge",
     ["recommendation systems", "recsys"], ["machine-learning"],
     ["machine-learning", "vector-search"]),

    # --- databases ----------------------------------------------------------
    ("postgresql", "PostgreSQL", "knowledge",
     ["postgres"], ["databases"], ["sql"]),
    ("mysql", "MySQL", "knowledge",
     [], ["databases"], ["sql"]),
    ("mongodb", "MongoDB", "knowledge",
     ["mongo"], ["databases"], ["nosql"]),
    ("nosql", "NoSQL databases", "knowledge",
     ["nosql"], ["databases"], ["mongodb", "redis"]),
    ("redis", "Redis", "knowledge",
     [], ["databases"], ["nosql"]),
    ("neo4j", "Neo4j", "knowledge",
     ["cypher"], ["databases"], ["knowledge-graphs"]),
    ("elasticsearch", "Elasticsearch", "knowledge",
     ["elastic search", "opensearch"], ["databases"], ["vector-search"]),

    # --- cloud / devops -----------------------------------------------------
    ("aws", "Amazon Web Services", "knowledge",
     ["aws", "amazon web services"], ["cloud-computing"], ["docker", "terraform"]),
    ("azure", "Microsoft Azure", "knowledge",
     ["azure"], ["cloud-computing"], ["docker"]),
    ("gcp", "Google Cloud Platform", "knowledge",
     ["gcp", "google cloud"], ["cloud-computing"], ["docker"]),
    ("docker", "Docker", "knowledge",
     ["containerisation", "containerization", "containers"], ["devops"],
     ["kubernetes", "ci-cd"]),
    ("kubernetes", "Kubernetes", "knowledge",
     ["k8s"], ["devops"], ["docker", "microservices"]),
    ("ci-cd", "CI/CD pipelines", "skill",
     ["ci/cd", "continuous integration", "continuous delivery", "jenkins"],
     ["devops"], ["docker", "git"]),
    ("terraform", "Terraform", "knowledge",
     ["infrastructure as code", "iac"], ["devops"], ["aws"]),
    ("linux", "Linux administration", "skill",
     ["linux", "unix", "shell scripting", "bash"], ["devops"], ["docker"]),
    ("git", "Git version control", "skill",
     ["git", "version control", "github"], ["programming"], ["ci-cd"]),
    ("monitoring", "system monitoring", "skill",
     ["observability", "prometheus", "grafana"], ["devops"], ["linux"]),

    # --- security -----------------------------------------------------------
    ("application-security", "application security", "skill",
     ["appsec", "secure coding"], ["cybersecurity"], ["cryptography"]),
    ("cryptography", "cryptography", "knowledge",
     ["encryption"], ["cybersecurity"], ["application-security"]),
    ("penetration-testing", "penetration testing", "skill",
     ["pentesting", "ethical hacking"], ["cybersecurity"], ["application-security"]),
    ("iam", "identity and access management", "skill",
     ["iam", "authentication", "oauth"], ["cybersecurity"], ["application-security"]),

    # --- mobile -------------------------------------------------------------
    ("android", "Android development", "knowledge",
     ["android"], ["mobile-development"], ["kotlin", "java"]),
    ("kotlin", "Kotlin", "knowledge",
     [], ["programming"], ["android", "java"]),
    ("ios", "iOS development", "knowledge",
     ["ios", "swift"], ["mobile-development"], []),
    ("flutter", "Flutter", "knowledge",
     ["dart"], ["mobile-development"], ["react-native"]),
    ("react-native", "React Native", "knowledge",
     ["react native"], ["mobile-development"], ["react", "flutter"]),

    # --- design -------------------------------------------------------------
    ("ui-design", "user interface design", "skill",
     ["ui design", "interface design"], ["product-design"], ["figma", "design-systems"]),
    ("ux-research", "user experience research", "skill",
     ["ux research", "user research", "usability testing"], ["product-design"],
     ["ui-design"]),
    ("figma", "Figma", "knowledge",
     [], ["product-design"], ["ui-design"]),
    ("design-systems", "design systems", "knowledge",
     ["design system"], ["product-design"], ["ui-design", "css"]),
    ("prototyping", "prototyping", "skill",
     ["rapid prototyping", "wireframing"], ["product-design"], ["figma"]),

    # --- testing ------------------------------------------------------------
    ("unit-testing", "unit testing", "skill",
     ["unit tests"], ["software-testing"], ["test-automation"]),
    ("test-automation", "test automation", "skill",
     ["automated testing"], ["software-testing"], ["selenium", "ci-cd"]),
    ("selenium", "Selenium", "knowledge",
     [], ["software-testing"], ["test-automation"]),
    ("performance-testing", "performance testing", "skill",
     ["load testing", "stress testing"], ["software-testing"], ["monitoring"]),

    # --- professional / transversal ----------------------------------------
    ("project-management", "project management", "skill",
     ["managing projects"], [], ["agile", "stakeholder-communication"]),
    ("agile", "agile methodologies", "knowledge",
     ["agile", "scrum", "kanban"], [], ["project-management"]),
    ("stakeholder-communication", "stakeholder communication", "skill",
     ["stakeholder management", "client communication"], [],
     ["technical-writing", "project-management"]),
    ("technical-writing", "technical writing", "skill",
     ["documentation", "technical documentation"], [], ["stakeholder-communication"]),
    ("team-leadership", "team leadership", "skill",
     ["leading teams", "people management", "mentoring"], [], ["project-management"]),
    ("requirements-analysis", "requirements analysis", "skill",
     ["requirements gathering", "business analysis"], [],
     ["stakeholder-communication", "agile"]),
    ("data-storytelling", "data storytelling", "skill",
     ["presenting data", "insight communication"], [],
     ["data-visualisation", "stakeholder-communication"]),
    ("problem-solving", "analytical problem solving", "skill",
     ["problem solving", "analytical thinking"], [], []),
]


# ---------------------------------------------------------------------------
# Occupations: (slug, label, essential, optional, description)
# ---------------------------------------------------------------------------

OCCUPATIONS: List[Tuple[str, str, List[str], List[str], str]] = [
    ("data-scientist", "data scientist",
     ["python", "statistics", "machine-learning", "data-analysis", "sql"],
     ["deep-learning", "data-visualisation", "scikit-learn", "data-storytelling"],
     "Applies statistical and machine learning methods to derive insight from data."),
    ("ml-engineer", "machine learning engineer",
     ["python", "machine-learning", "deep-learning", "mlops", "model-evaluation"],
     ["pytorch", "docker", "aws", "feature-engineering"],
     "Builds, trains and deploys machine learning models in production."),
    ("nlp-engineer", "NLP engineer",
     ["python", "nlp", "deep-learning", "machine-learning"],
     ["llm", "pytorch", "rag", "model-evaluation"],
     "Develops language technology systems and text understanding pipelines."),
    ("ai-research-engineer", "AI research engineer",
     ["python", "deep-learning", "machine-learning", "model-evaluation"],
     ["llm", "rag", "knowledge-graphs", "pytorch", "technical-writing"],
     "Prototypes and evaluates novel AI methods and publishes findings."),
    ("data-engineer", "data engineer",
     ["sql", "python", "etl", "data-warehousing"],
     ["spark", "airflow", "kafka", "aws", "dbt"],
     "Designs and operates data pipelines and warehouses."),
    ("backend-engineer", "backend engineer",
     ["rest-api", "databases", "git"],
     ["python", "java", "docker", "microservices", "postgresql"],
     "Builds server-side services and APIs."),
    ("frontend-engineer", "frontend engineer",
     ["javascript", "html", "css", "react"],
     ["typescript", "nextjs", "web-accessibility", "tailwind"],
     "Builds user-facing web interfaces."),
    ("fullstack-engineer", "full stack engineer",
     ["javascript", "rest-api", "databases", "git"],
     ["react", "nodejs", "docker", "postgresql", "typescript"],
     "Works across both client and server layers of web applications."),
    ("devops-engineer", "DevOps engineer",
     ["linux", "ci-cd", "docker", "cloud-computing"],
     ["kubernetes", "terraform", "monitoring", "aws"],
     "Automates build, deployment and infrastructure operations."),
    ("mobile-developer", "mobile application developer",
     ["mobile-development", "git"],
     ["android", "ios", "flutter", "react-native", "rest-api"],
     "Develops applications for mobile platforms."),
    ("qa-engineer", "QA automation engineer",
     ["software-testing", "test-automation", "git"],
     ["selenium", "ci-cd", "python", "performance-testing"],
     "Designs and automates software quality assurance processes."),
    ("ux-designer", "UX designer",
     ["ux-research", "ui-design", "prototyping"],
     ["figma", "design-systems", "web-accessibility"],
     "Researches user needs and designs product experiences."),
    ("security-engineer", "security engineer",
     ["cybersecurity", "application-security", "linux"],
     ["cryptography", "penetration-testing", "iam", "cloud-computing"],
     "Protects systems and applications from security threats."),
    ("business-analyst", "business analyst",
     ["requirements-analysis", "stakeholder-communication", "data-analysis"],
     ["sql", "data-visualisation", "agile", "data-storytelling"],
     "Translates business needs into system and data requirements."),
    ("analytics-engineer", "analytics engineer",
     ["sql", "data-warehousing", "data-analysis"],
     ["dbt", "python", "data-visualisation", "etl"],
     "Models warehouse data into reliable analytics-ready datasets."),
]


# ---------------------------------------------------------------------------
# Job postings: (id, title, occupation, company, location, seniority,
#                extra_essential, extra_optional, description)
# ---------------------------------------------------------------------------

JOBS: List[Tuple[str, str, str, str, str, str, List[str], List[str], str]] = [
    ("J001", "Junior Data Scientist", "data-scientist", "Northwind Analytics",
     "Pune, IN", "junior", [], ["r-lang"],
     "Support the analytics team in building predictive models and reporting on "
     "customer behaviour. You will work with product and marketing stakeholders."),
    ("J002", "Machine Learning Engineer", "ml-engineer", "Vertex Labs",
     "Bengaluru, IN", "mid", ["docker"], ["kubernetes", "aws"],
     "Own the training and deployment lifecycle for recommendation and ranking "
     "models serving millions of daily requests."),
    ("J003", "NLP Engineer, Search", "nlp-engineer", "Lumen Search",
     "Remote, IN", "mid", ["vector-search"], ["elasticsearch", "rag"],
     "Improve semantic search relevance using language models and embedding "
     "retrieval across a large multilingual catalogue."),
    ("J004", "AI Research Engineer, Applied LLM", "ai-research-engineer",
     "Meridian AI", "Hyderabad, IN", "senior", ["llm"],
     ["rag", "knowledge-graphs", "technical-writing"],
     "Prototype retrieval-augmented reasoning systems and publish results. "
     "Works closely with product to move research into production."),
    ("J005", "Data Engineer", "data-engineer", "Cobalt Systems",
     "Nagpur, IN", "mid", [], ["spark", "airflow"],
     "Build and maintain batch and streaming pipelines feeding the analytics "
     "warehouse."),
    ("J006", "Senior Data Engineer", "data-engineer", "Cobalt Systems",
     "Nagpur, IN", "senior", ["spark"], ["kafka", "dbt", "team-leadership"],
     "Lead the pipeline platform team and set data modelling standards."),
    ("J007", "Backend Engineer, Payments", "backend-engineer", "Fintrail",
     "Mumbai, IN", "mid", ["python"], ["postgresql", "microservices"],
     "Develop resilient payment services with strong consistency requirements."),
    ("J008", "Backend Engineer, Java", "backend-engineer", "Orion Retail",
     "Pune, IN", "mid", ["java", "spring-boot"], ["kafka", "microservices"],
     "Build order management services for a high-volume retail platform."),
    ("J009", "Frontend Engineer", "frontend-engineer", "Brightline",
     "Remote, IN", "junior", [], ["tailwind", "typescript"],
     "Build accessible, responsive interfaces for our customer dashboard."),
    ("J010", "Senior Frontend Engineer", "frontend-engineer", "Brightline",
     "Remote, IN", "senior", ["typescript"], ["nextjs", "design-systems", "team-leadership"],
     "Lead the design system effort and mentor a team of three engineers."),
    ("J011", "Full Stack Engineer", "fullstack-engineer", "Kestrel Software",
     "Nagpur, IN", "mid", ["react", "nodejs"], ["docker", "postgresql"],
     "Work end to end on features for a B2B scheduling product."),
    ("J012", "DevOps Engineer", "devops-engineer", "Cobalt Systems",
     "Pune, IN", "mid", ["kubernetes"], ["terraform", "monitoring"],
     "Automate deployment pipelines and manage container infrastructure."),
    ("J013", "Site Reliability Engineer", "devops-engineer", "Vertex Labs",
     "Bengaluru, IN", "senior", ["monitoring", "kubernetes"], ["terraform", "golang"],
     "Own service reliability, incident response and capacity planning."),
    ("J014", "Android Developer", "mobile-developer", "Nimbus Apps",
     "Nagpur, IN", "junior", ["android"], ["kotlin", "rest-api"],
     "Build and ship features for a consumer Android application."),
    ("J015", "Cross-platform Mobile Developer", "mobile-developer", "Nimbus Apps",
     "Remote, IN", "mid", ["flutter"], ["react-native", "rest-api"],
     "Develop a shared codebase serving both Android and iOS users."),
    ("J016", "QA Automation Engineer", "qa-engineer", "Orion Retail",
     "Pune, IN", "mid", ["selenium"], ["python", "ci-cd"],
     "Expand automated regression coverage across web and API layers."),
    ("J017", "UX Designer", "ux-designer", "Brightline",
     "Remote, IN", "mid", ["figma"], ["design-systems", "web-accessibility"],
     "Run discovery research and design flows for the analytics product."),
    ("J018", "Application Security Engineer", "security-engineer", "Fintrail",
     "Mumbai, IN", "senior", ["application-security"], ["cryptography", "iam"],
     "Lead secure development practices and threat modelling for payment systems."),
    ("J019", "Business Analyst, Data Products", "business-analyst", "Northwind Analytics",
     "Pune, IN", "mid", ["sql"], ["data-visualisation", "agile"],
     "Bridge business stakeholders and the data platform team."),
    ("J020", "Analytics Engineer", "analytics-engineer", "Northwind Analytics",
     "Remote, IN", "mid", ["dbt"], ["python", "data-visualisation"],
     "Model raw warehouse tables into trusted analytics datasets."),
    ("J021", "Junior Backend Engineer", "backend-engineer", "Kestrel Software",
     "Nagpur, IN", "junior", ["python"], ["flask", "postgresql"],
     "Support API development for our scheduling platform under mentorship."),
    ("J022", "Machine Learning Engineer, Recommendations", "ml-engineer",
     "Lumen Search", "Remote, IN", "senior", ["recommender-systems"],
     ["vector-search", "spark", "aws"],
     "Design and scale the ranking stack behind personalised recommendations."),
    ("J023", "Data Scientist, Marketing", "data-scientist", "Orion Retail",
     "Mumbai, IN", "mid", ["data-visualisation"], ["r-lang", "data-storytelling"],
     "Measure campaign effectiveness and build customer segmentation models."),
    ("J024", "Graduate Software Engineer", "backend-engineer", "Meridian AI",
     "Hyderabad, IN", "junior", [], ["python", "git", "docker"],
     "Rotational graduate programme across backend and platform teams."),
]


# ---------------------------------------------------------------------------
# Sample resumes used by the UI and the evaluation harness
# ---------------------------------------------------------------------------

RESUMES = [
    {
        "candidate_id": "C001",
        "name": "Final-year CS student, ML focus",
        "text": """PRIYA SHARMA
Final-year B.E. Computer Science student | Nagpur, India

SUMMARY
Final-year engineering student focused on machine learning and data analysis.
Comfortable with Python and SQL, currently learning deep learning.

SKILLS
Python, SQL, pandas, NumPy, scikit-learn, statistics, data analysis,
data visualization, Git, problem solving

PROJECTS
Customer Churn Prediction - Built a classification model in scikit-learn on a
telecom dataset, performed feature engineering and model evaluation, reported
results to the department review panel.
Sales Dashboard - Analysed two years of retail sales data with pandas and
produced an interactive dashboard for the college business club.

EDUCATION
B.E. Computer Science, 2022-2026

EXPERIENCE
Data Analytics Intern, 6 months - Cleaned and analysed survey data, wrote SQL
queries against the reporting database, presented findings to stakeholders.
""",
    },
    {
        "candidate_id": "C002",
        "name": "Frontend developer, 3 years",
        "text": """ARJUN MEHTA
Frontend Developer | Remote, India

SUMMARY
Frontend developer with 3 years of experience building production React
applications. Strong CSS and accessibility background.

SKILLS
JavaScript, TypeScript, React, HTML, CSS, Tailwind CSS, Git, web accessibility,
unit testing, stakeholder communication

EXPERIENCE
Frontend Developer, 3 years - Built and maintained a customer dashboard in
React and TypeScript. Led the accessibility audit that brought the product to
WCAG AA. Worked with designers on the component library.

EDUCATION
B.Tech Information Technology
""",
    },
    {
        "candidate_id": "C003",
        "name": "Backend engineer moving to data",
        "text": """RAHUL VERMA
Backend Engineer | Pune, India

SUMMARY
Backend engineer with 4 years of experience in Python services, now moving
towards data engineering.

SKILLS
Python, Flask, REST API design, PostgreSQL, SQL, Docker, Git, Linux,
CI/CD pipelines, microservices architecture

EXPERIENCE
Backend Engineer, 4 years - Designed REST APIs for an inventory platform.
Containerised services with Docker and set up CI/CD pipelines. Optimised
PostgreSQL queries handling significant daily volume.

No experience with Kubernetes yet.

EDUCATION
B.E. Computer Engineering
""",
    },
    {
        "candidate_id": "C004",
        "name": "Applied AI engineer, LLM and graphs",
        "text": """SNEHA IYER
Applied AI Engineer | Hyderabad, India

SUMMARY
Applied AI engineer with 5 years of experience. Recent work centres on large
language models, retrieval-augmented generation and knowledge graphs.

SKILLS
Python, PyTorch, deep learning, natural language processing, large language
models, retrieval-augmented generation, knowledge graphs, Neo4j, vector
similarity search, model evaluation, technical writing, Docker

EXPERIENCE
Applied AI Engineer, 5 years - Built a retrieval-augmented question answering
system over an internal knowledge graph. Evaluated language model outputs for
factual grounding. Published two workshop papers.

EDUCATION
M.Tech Artificial Intelligence
""",
    },
]


def build() -> Dict[str, object]:
    skills = []
    for slug, label, stype, alts, broader, related in SKILLS:
        skills.append(
            {
                "uri": uri(slug),
                "label": label,
                "skill_type": stype,
                "alt_labels": alts,
                "broader": [uri(b) for b in broader],
                "related": [uri(r) for r in related],
                "description": "",
            }
        )

    occupations = []
    for slug, label, ess, opt, desc in OCCUPATIONS:
        occupations.append(
            {
                "uri": occ_uri(slug),
                "label": label,
                "essential_skills": [uri(s) for s in ess],
                "optional_skills": [uri(s) for s in opt],
                "description": desc,
            }
        )

    occ_index = {o[0]: o for o in OCCUPATIONS}
    jobs = []
    for jid, title, occ_slug, company, loc, sen, x_ess, x_opt, desc in JOBS:
        base = occ_index[occ_slug]
        ess = list(dict.fromkeys(list(base[2]) + list(x_ess)))
        opt = [s for s in dict.fromkeys(list(base[3]) + list(x_opt)) if s not in ess]
        jobs.append(
            {
                "job_id": jid,
                "title": title,
                "company": company,
                "location": loc,
                "seniority": sen,
                "description": desc,
                "occupation_uri": occ_uri(occ_slug),
                "essential_skills": [uri(s) for s in ess],
                "optional_skills": [uri(s) for s in opt],
            }
        )

    return {"skills": skills, "occupations": occupations, "jobs": jobs, "resumes": RESUMES}


def main() -> None:
    data = build()
    validate(data)
    with open(os.path.join(HERE, "taxonomy.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"skills": data["skills"], "occupations": data["occupations"]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(os.path.join(HERE, "jobs.json"), "w", encoding="utf-8") as f:
        json.dump(data["jobs"], f, indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, "resumes.json"), "w", encoding="utf-8") as f:
        json.dump(data["resumes"], f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {len(data['skills'])} skills, {len(data['occupations'])} occupations, "
        f"{len(data['jobs'])} jobs, {len(data['resumes'])} resumes."
    )


def validate(data: Dict[str, object]) -> None:
    """Fail loudly if the seed data is internally inconsistent."""
    skill_uris = {s["uri"] for s in data["skills"]}  # type: ignore[index]
    if len(skill_uris) != len(data["skills"]):       # type: ignore[arg-type]
        raise ValueError("duplicate skill URIs in seed taxonomy")

    for s in data["skills"]:                         # type: ignore[union-attr]
        for b in s["broader"]:
            if b not in skill_uris:
                raise ValueError(f"{s['uri']} has unknown broader {b}")
        for r in s["related"]:
            if r not in skill_uris:
                raise ValueError(f"{s['uri']} has unknown related {r}")
        if s["uri"] in s["broader"] or s["uri"] in s["related"]:
            raise ValueError(f"{s['uri']} links to itself")

    for o in data["occupations"]:                    # type: ignore[union-attr]
        for u in list(o["essential_skills"]) + list(o["optional_skills"]):
            if u not in skill_uris:
                raise ValueError(f"occupation {o['uri']} has unknown skill {u}")
        overlap = set(o["essential_skills"]) & set(o["optional_skills"])
        if overlap:
            raise ValueError(f"occupation {o['uri']} lists {overlap} as both essential and optional")

    seen_jobs = set()
    for j in data["jobs"]:                           # type: ignore[union-attr]
        if j["job_id"] in seen_jobs:
            raise ValueError(f"duplicate job id {j['job_id']}")
        seen_jobs.add(j["job_id"])
        for u in list(j["essential_skills"]) + list(j["optional_skills"]):
            if u not in skill_uris:
                raise ValueError(f"job {j['job_id']} has unknown skill {u}")
        overlap = set(j["essential_skills"]) & set(j["optional_skills"])
        if overlap:
            raise ValueError(f"job {j['job_id']} lists {overlap} as both essential and optional")


if __name__ == "__main__":
    main()
